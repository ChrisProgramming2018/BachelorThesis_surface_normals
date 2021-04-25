from replay_buffer_depth import ReplayBufferDepth
   
size= 256
memory_real = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 7501, "cuda")
memory_sim = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 7501, "cuda")
print("load buffer ..")
#memory_real.load_memory_normals("../data/real_world_normal_buffer-valid")
#memory_sim.load_memory_normals("../data/normal_sim_memory-valid")
memory_real.load_memory_normals("../data/real_world_normal_buffer-train")
memory_sim.load_memory_normals("../data/normal_sim_memory-train")
print("real size", memory_real.idx)
print("sim size", memory_sim.idx)


memory_simreal = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 15001, "cuda")

for i in range(memory_real.idx):
    print(memory_simreal.idx)
    memory_simreal.depth[memory_real.idx] = memory_real.depth[i]
    memory_simreal.normals[memory_simreal.idx] = memory_real.normals[i]
    memory_simreal.obses[memory_simreal.idx] = memory_real.obses[i]
    memory_simreal.idx +=1
print("size sim_real buffer ", memory_simreal.idx)
start = memory_simreal.idx
for i in range(memory_sim.idx):
    print(memory_simreal.idx, i)
    memory_simreal.depth[memory_simreal.idx] = memory_sim.depth[i]
    memory_simreal.normals[memory_simreal.idx] = memory_sim.normals[i]
    memory_simreal.obses[memory_simreal.idx] = memory_sim.obses[i]
    memory_simreal.idx += 1

print("size sim_real buffer ", memory_simreal.idx)
#memory_simreal.save_memory_normals("sim_real_buffer-valid")
memory_simreal.save_memory_normals("sim_real_buffer-train")
