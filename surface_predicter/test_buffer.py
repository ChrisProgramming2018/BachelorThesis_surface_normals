from replay_buffer_depth import ReplayBufferDepth

size= 256
create = False
#memory = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 51, "cuda")
#memory_small = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 2001, "cuda")
memory_small = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 15001, "cuda")
#memory.load_memory("depth_memory5k")
#memory.load_memory("real_world_depth_buffer-train")"
print("load buffer ..")
if create:
    memory_small.load_memory("small_depth_buffer")
else:
    #memory_small.load_memory_normals("../data/sim_real_buffer-valid")
    memory_small.load_memory_normals("../data/sim_real_buffer-train")

print("size small buffer ", memory_small.idx)

if create:
    memory_small.create_surface_normals("small_normal_buffer")

memory_small.test_surface_normals(10142)
#memory.test_surface_normals(49)
