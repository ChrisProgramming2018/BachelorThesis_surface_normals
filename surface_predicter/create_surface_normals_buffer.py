from replay_buffer_depth import ReplayBufferDepth

size= 256
memory = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 2001, "cuda")
print("load buffer ..")
#memory.load_memory("real_world_depth_buffer-train")
memory.load_memory("real_world_depth_buffer-valid")
print("train  buffer ", memory.idx)

memory.create_surface_normals("real_world_normal_buffer-valid")

#memory_small.test_surface_normals(42)
#memory.test_surface_normals(49)
