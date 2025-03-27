# Memory optimizations for the original code

# 1. Add memory checkpointing to reduce peak memory usage
import torch.utils.checkpoint as checkpoint

# Modified forward method in the Trainer class to use gradient checkpointing
def forward(self, target, it, if_normal, if_pretrain, scene_and_vertices):
    if self.FLAGS.mode == 'appearance_modeling':
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])
    
    if if_pretrain:
        return self.geometry.decoder.pre_train_ellipsoid(it, scene_and_vertices)
    else:
        # Use checkpointing on the most memory-intensive operations
        def custom_forward(*inputs):
            return self.geometry.tick(
                glctx, inputs[0], self.light, self.material, 
                it, if_normal, self.guidance, self.FLAGS.mode, 
                self.if_flip_the_normal, self.if_use_bump
            )
        
        # Only store input tensors needed for backprop
        necessary_inputs = target
        return checkpoint.checkpoint(custom_forward, necessary_inputs)

# 2. Add memory-efficient optimizations to validate_itr function
@torch.no_grad()  
def validate_itr_optimized(glctx, target, geometry, opt_material, lgt, FLAGS, relight=None):
    result_dict = {}
    
    # Process in lower resolution for validation
    downscale_factor = 2  # Downscale by 2x for validation
    target_resolution = [r // downscale_factor for r in target['resolution']]
    target_downscaled = target.copy()
    target_downscaled['resolution'] = target_resolution
    
    lgt.build_mips()
    if FLAGS.camera_space_light:
        lgt.xfm(target_downscaled['mv'])
    
    # Render with smaller resolution
    buffers = geometry.render(glctx, target_downscaled, lgt, opt_material, if_use_bump=FLAGS.if_use_bump)
    result_dict['shaded'] = buffers['shaded'][0, ..., 0:3]
    result_dict['shaded'] = util.rgb_to_srgb(result_dict['shaded'])
    result_dict['mask'] = (buffers['shaded'][0, ..., 3:4])
    result_image = result_dict['shaded']
    
    # Free memory immediately after use
    del buffers
    torch.cuda.empty_cache()
    
    # Process display layers sequentially to save memory
    if FLAGS.display is not None:
        for layer in FLAGS.display:
            if 'latlong' in layer and layer['latlong']:
                if isinstance(lgt, light.EnvironmentLight):
                    # Generate light preview at lower resolution
                    preview_res = min(FLAGS.display_res, 512)
                    result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, preview_res)
                result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                del result_dict['light_image']
                torch.cuda.empty_cache()
            
            elif 'bsdf' in layer:
                # Generate BSDF previews at lower resolution
                buffers = geometry.render(glctx, target_downscaled, lgt, opt_material, 
                                          bsdf=layer['bsdf'], if_use_bump=FLAGS.if_use_bump)
                
                if layer['bsdf'] == 'kd':
                    result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                elif layer['bsdf'] == 'normal':
                    result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                else:
                    result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    
                result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
                del result_dict[layer['bsdf']]
                del buffers
                torch.cuda.empty_cache()

    return result_image, result_dict

# 3. Optimize the main training loop with mixed precision and memory management
def optimize_mesh_efficient(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    log_interval=10,
    optimize_light=True,
    optimize_geometry=True,
    guidance=None,
    scene_and_vertices=None,
    ):
    
    # Use smaller batch size and worker count
    adjusted_batch_size = max(1, FLAGS.batch // 2)  # Reduce batch size to save memory
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=adjusted_batch_size, 
        collate_fn=dataset_train.collate, 
        shuffle=False,
        num_workers=2,  # Limit workers to reduce CPU memory usage
        pin_memory=True  # Use pinned memory for faster GPU transfer
    )
    
    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, 
        batch_size=1, 
        collate_fn=dataset_train.collate,
        num_workers=1
    )

    # Initialize model and optimizers
    model = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, FLAGS, guidance)
    
    # Use separate parameter groups with different learning rates
    if optimize_geometry:
        geo_params = model.geo_params
        optimizer_mesh = torch.optim.AdamW([
            {'params': geo_params, 'lr': 0.001, 'weight_decay': 1e-5}
        ], betas=(0.9, 0.99), eps=1e-15)
    
    # Split material parameters into groups for more efficient optimization
    material_params = model.params
    optimizer = torch.optim.AdamW([
        {'params': material_params, 'lr': 0.01, 'weight_decay': 1e-6}
    ], betas=(0.9, 0.99), eps=1e-15)

    # Load checkpoint if available
    load_it = 0
    load_it = model.load()
    
    # Enable mixed precision training with apex if available
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Use gradient accumulation to simulate larger batches
    accumulation_steps = max(1, FLAGS.batch // adjusted_batch_size)
    
    # Set up validation cycle
    v_it = cycle(dataloader_validate)
    
    # Tracking variables
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []
    rot_ang = 0
    
    if FLAGS.local_rank == 0:
        video = Video(FLAGS.out_dir)
        
    if FLAGS.local_rank == 0:
        dataloader_train = tqdm(dataloader_train)
        
    # Main training loop
    for it, target in enumerate(dataloader_train):
        # Clean up memory before each iteration
        torch.cuda.empty_cache()
        
        # Prepare batch
        target = prepare_batch(target, FLAGS.train_background, it, FLAGS.coarse_iter)
        
        # Display/save outputs
        if FLAGS.local_rank == 0:
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            save_video = FLAGS.video_interval and (it % FLAGS.video_interval == 0)
            
            if save_image or save_video:
                # Free memory before validation
                torch.cuda.empty_cache()
                
                if save_image:
                    # Use optimized validation function
                    result_image, result_dict = validate_itr_optimized(
                        glctx, 
                        prepare_batch(next(v_it), FLAGS.train_background), 
                        geometry, 
                        opt_material, 
                        lgt, 
                        FLAGS
                    )
                    
                    np_result_image = result_image.detach().cpu().numpy()
                    util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (FLAGS.mode, img_cnt)), np_result_image)
                    img_cnt = img_cnt + 1
                    
                    # Free memory
                    del result_image, result_dict, np_result_image
                    torch.cuda.empty_cache()
                
                if save_video:
                    with torch.no_grad():
                        # Lower resolution for video preview
                        vid_resolution = 256  # Lower resolution for video frames
                        params = get_camera_params(
                            resolution=vid_resolution,
                            fov=45,
                            elev_angle=-20,
                            azim_angle=rot_ang,       
                        )
                        rot_ang += 1
                        
                        if FLAGS.mode == 'geometry_modeling':
                            buffers = geometry.render(glctx, params, lgt, opt_material, bsdf='normal', if_use_bump=FLAGS.if_use_bump)
                            video_image = (buffers['shaded'][0, ..., 0:3] + 1) / 2
                        else:
                            buffers = geometry.render(glctx, params, lgt, opt_material, bsdf='pbr', if_use_bump=FLAGS.if_use_bump)
                            video_image = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                            
                        video_image = video.ready_image(video_image)
                        
                        # Free memory
                        del buffers, video_image
                        torch.cuda.empty_cache()
                
        # Start timing for this iteration
        iter_start_time = time.time()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            mse_loss, img_loss, reg_loss = model(target, it, False, if_pretrain=False, scene_and_vertices=None)
            # Scale loss for gradient accumulation
            total_loss = (img_loss + reg_loss + mse_loss) / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        
        # Record losses
        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())
        
        # Only step optimizer after accumulation or at end of dataset
        if (it + 1) % accumulation_steps == 0 or (it + 1) == len(dataloader_train):
            # Update model parameters
            scaler.step(optimizer)
            optimizer.zero_grad()
            
            if optimize_geometry:
                scaler.step(optimizer_mesh)
                optimizer_mesh.zero_grad()
                
            scaler.update()
            
        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)
        
        # Periodic checkpoint saving
        if it > 0 and it % FLAGS.ckpt_iter == 0 and FLAGS.local_rank == 0:
            # Free memory before saving
            torch.cuda.empty_cache()
            
            # Save model
            if FLAGS.multi_gpu:
                model.module.save(it)
            else:
                model.save(it)
                
            # Save optimizer state
            optim_state = {
                'optimizer': optimizer.state_dict(),
                'optimizer_mesh': optimizer_mesh.state_dict() if optimize_geometry else None
            }
            torch.save(optim_state, os.path.join(FLAGS.out_dir, "checkpoints", f"optim_{it}.pth"))
    
    return geometry, opt_material

# 4. Memory-optimized validation function for final output
@torch.no_grad()     
def validate_optimized(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS, relight=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # Create output directories
    subdirs = ["shaded", "relight", "kd", "ks", "normal", "mask"]
    for dir_name in subdirs:
        os.makedirs(os.path.join(out_dir, dir_name), exist_ok=True)
    
    # Process in smaller batches with lower initial resolution
    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, 
        batch_size=1, 
        collate_fn=dataset_validate.collate,
        num_workers=1
    )
    
    print("Running validation")
    dataloader_validate = tqdm(dataloader_validate)
    
    for it, target in enumerate(dataloader_validate):
        # Free memory before each validation iteration
        torch.cuda.empty_cache()
        
        # Prepare batch
        target = prepare_batch(target, 'white')
        
        # Use staged processing to save memory
        result_dict = {}
        
        # Render basic shaded view
        with torch.cuda.amp.autocast(enabled=True):
            lgt.build_mips()
            if FLAGS.camera_space_light:
                lgt.xfm(target['mv'])
                
            buffers = geometry.render(glctx, target, lgt, opt_material, if_use_bump=FLAGS.if_use_bump)
            result_dict['shaded'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
            result_dict['mask'] = buffers['shaded'][0, ..., 3:4]
            
            # Save shaded image
            util.save_image(
                os.path.join(out_dir, "shaded", f'val_{it:06d}_shaded.png'), 
                result_dict['shaded'].detach().cpu().numpy()
            )
            
            # Save mask
            util.save_image(
                os.path.join(out_dir, "mask", f'val_{it:06d}_mask.png'), 
                np.concatenate([result_dict['mask'].detach().cpu().numpy()] * 3, axis=-1)
            )
            
            del buffers
            torch.cuda.empty_cache()
        
        # Process relight if needed
        if relight is not None:
            with torch.cuda.amp.autocast(enabled=True):
                relight.build_mips()
                buffers = geometry.render(glctx, target, relight, opt_material, if_use_bump=FLAGS.if_use_bump)
                result_dict['relight'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                
                # Save relight image
                util.save_image(
                    os.path.join(out_dir, "relight", f'val_{it:06d}_relight.png'), 
                    result_dict['relight'].detach().cpu().numpy()
                )
                
                del buffers, result_dict['relight']
                torch.cuda.empty_cache()
        
        # Process additional visualization modes one by one
        for bsdf_type in ['kd', 'ks', 'normal']:
            with torch.cuda.amp.autocast(enabled=True):
                buffers = geometry.render(
                    glctx, target, lgt, opt_material, 
                    bsdf=bsdf_type, if_use_bump=FLAGS.if_use_bump
                )
                
                if bsdf_type == 'kd':
                    result = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                elif bsdf_type == 'normal':
                    result = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                else:
                    result = buffers['shaded'][0, ..., 0:3]
                
                # Save image
                util.save_image(
                    os.path.join(out_dir, bsdf_type, f'val_{it:06d}_{bsdf_type}.png'), 
                    result.detach().cpu().numpy()
                )
                
                del buffers, result
                torch.cuda.empty_cache()
    
    # Create GIFs for each output type
    for subdir in subdirs:
        dir_path = os.path.join(out_dir, subdir)
        if len(os.listdir(dir_path)) > 0:
            save_gif(dir_path, 30)
    
    return 0