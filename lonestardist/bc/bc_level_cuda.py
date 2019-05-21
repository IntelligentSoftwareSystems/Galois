from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("bc_level_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality'), ('float *', 'p_dependency'), ('ShortPathType *', 'p_num_shortest_paths')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_betweeness_centrality[src] = 0"]),
CBlock(["p_num_shortest_paths[src]    = 0"]),
CBlock(["p_dependency[src]            = 0"]),
]),
]),
]),
Kernel("InitializeIteration", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint64_t ', 'local_current_src_node'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('float *', 'p_dependency'), ('ShortPathType *', 'p_num_shortest_paths')],
[
CDecl([("bool", "is_source", "")]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["is_source = graph.node_data[src] == local_current_src_node"]),
If("!is_source",
[
CBlock(["p_current_length[src]     = local_infinity"]),
CBlock(["p_num_shortest_paths[src] = 0"]),
],
[
CBlock(["p_current_length[src]     = 0"]),
CBlock(["p_num_shortest_paths[src] = 1"]),
]),
CBlock(["p_dependency[src]       = 0"]),
]),
]),
]),
Kernel("ForwardPass", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t', 'local_r'), ('uint32_t *', 'p_current_length'), ('ShortPathType *', 'p_num_shortest_paths'), ('DynamicBitset&', 'bitset_current_length'), ('DynamicBitset&', 'bitset_num_shortest_paths'), ('HGAccumulator<uint32_t>', 'dga')],
[
CDecl([("__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage", "dga_ts", "")]),
CBlock(["dga.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] == local_r",
[
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = 1 + p_current_length[src]"]),
CDecl([("uint32_t", "old", "")]),
CBlock(["old = atomicTestMin(&p_current_length[dst], new_dist)"]),
If("old > new_dist",
[
CBlock(["bitset_current_length.set(dst)"]),
CDecl([("double", "nsp", "")]),
CBlock(["nsp = p_num_shortest_paths[src]"]),
CBlock(["atomicTestAdd(&p_num_shortest_paths[dst], nsp)"]),
CBlock(["bitset_num_shortest_paths.set(dst)"]),
CBlock(["dga.reduce( 1)"]),
],
[
If("old == new_dist",
[
CDecl([("double", "nsp", "")]),
CBlock(["nsp = p_num_shortest_paths[src]"]),
CBlock(["atomicTestAdd(&p_num_shortest_paths[dst], nsp)"]),
CBlock(["bitset_num_shortest_paths.set(dst)"]),
CBlock(["dga.reduce( 1)"]),
]),
]),
]),
),
]),
CBlock(["dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(dga_ts)"], parse = False),
]),
Kernel("MiddleSync", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('DynamicBitset&', 'bitset_num_shortest_paths')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
CBlock(["bitset_num_shortest_paths.set(src)"]),
]),
]),
]),
]),
Kernel("BackwardPass", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t', 'local_r'), ('uint32_t *', 'p_current_length'), ('float *', 'p_dependency'), ('ShortPathType *', 'p_num_shortest_paths'), ('DynamicBitset&', 'bitset_dependency')],
[
CDecl([("uint32_t", "dest_to_find", "")]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] == local_r",
[
CBlock(["dest_to_find = p_current_length[src] + 1"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
If("dest_to_find == p_current_length[dst]",
[
CDecl([("float", "contrib", "")]),
CBlock(["contrib = ((float)1 + p_dependency[dst]) / p_num_shortest_paths[dst]"]),
CBlock(["p_dependency[src] = p_dependency[src] + contrib"]),
CBlock(["bitset_dependency.set(src)"]),
]),
]),
),
CBlock(["p_dependency[src] *= p_num_shortest_paths[src]"]),
]),
]),
Kernel("BC", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality'), ('float *', 'p_dependency')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_dependency[src] > 0",
[
CBlock(["p_betweeness_centrality[src] += p_dependency[src]"]),
]),
]),
]),
]),
Kernel("Sanity", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality'), ('HGAccumulator<float>', 'DGAccumulator_sum'), ('HGReduceMax<float>', 'DGAccumulator_max'), ('HGReduceMin<float>', 'DGAccumulator_min')],
[
CDecl([("__shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage", "DGAccumulator_sum_ts", "")]),
CBlock(["DGAccumulator_sum.thread_entry()"]),
CDecl([("__shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage", "DGAccumulator_max_ts", "")]),
CBlock(["DGAccumulator_max.thread_entry()"]),
CDecl([("__shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage", "DGAccumulator_min_ts", "")]),
CBlock(["DGAccumulator_min.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["DGAccumulator_max.reduce(p_betweeness_centrality[src])"]),
CBlock(["DGAccumulator_min.reduce(p_betweeness_centrality[src])"]),
CBlock(["DGAccumulator_sum.reduce( p_betweeness_centrality[src])"]),
]),
]),
CBlock(["DGAccumulator_sum.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_sum_ts)"], parse = False),
CBlock(["DGAccumulator_max.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_max_ts)"], parse = False),
CBlock(["DGAccumulator_min.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_min_ts)"], parse = False),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "__begin", "__end", "ctx->betweeness_centrality.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("InitializeGraph_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("InitializeGraph_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("InitializeIteration_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeIteration", ("ctx->gg", "__begin", "__end", "local_current_src_node", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeIteration_allNodes_cuda", [('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeIteration_cuda(0, ctx->gg.nnodes, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("InitializeIteration_masterNodes_cuda", [('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeIteration_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("InitializeIteration_nodesWithEdges_cuda", [('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeIteration_cuda(0, ctx->numNodesWithEdges, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("ForwardPass_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint32_t &', 'dga'), ('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint32_t>", "dgaval", " = Shared<uint32_t>(1)")]),
CDecl([("HGAccumulator<uint32_t>", "_dga", "")]),
CBlock(["*(dgaval.cpu_wr_ptr()) = 0"]),
CBlock(["_dga.rv = dgaval.gpu_wr_ptr()"]),
Invoke("ForwardPass", ("ctx->gg", "__begin", "__end", "local_r", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "*(ctx->current_length.is_updated.gpu_rd_ptr())", "*(ctx->num_shortest_paths.is_updated.gpu_rd_ptr())", "_dga")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["dga = *(dgaval.cpu_rd_ptr())"]),
], host = True),
Kernel("ForwardPass_allNodes_cuda", [('uint32_t &', 'dga'), ('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ForwardPass_cuda(0, ctx->gg.nnodes, dga, local_r, ctx)"]),
], host = True),
Kernel("ForwardPass_masterNodes_cuda", [('uint32_t &', 'dga'), ('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ForwardPass_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, dga, local_r, ctx)"]),
], host = True),
Kernel("ForwardPass_nodesWithEdges_cuda", [('uint32_t &', 'dga'), ('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ForwardPass_cuda(0, ctx->numNodesWithEdges, dga, local_r, ctx)"]),
], host = True),
Kernel("MiddleSync_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("MiddleSync", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "*(ctx->num_shortest_paths.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("MiddleSync_allNodes_cuda", [('const uint32_t', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["MiddleSync_cuda(0, ctx->gg.nnodes, local_infinity, ctx)"]),
], host = True),
Kernel("MiddleSync_masterNodes_cuda", [('const uint32_t', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["MiddleSync_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx)"]),
], host = True),
Kernel("MiddleSync_nodesWithEdges_cuda", [('const uint32_t', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["MiddleSync_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx)"]),
], host = True),
Kernel("BackwardPass_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("BackwardPass", ("ctx->gg", "__begin", "__end", "local_r", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "*(ctx->dependency.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("BackwardPass_allNodes_cuda", [('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["BackwardPass_cuda(0, ctx->gg.nnodes, local_r, ctx)"]),
], host = True),
Kernel("BackwardPass_masterNodes_cuda", [('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["BackwardPass_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_r, ctx)"]),
], host = True),
Kernel("BackwardPass_nodesWithEdges_cuda", [('uint32_t', 'local_r'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["BackwardPass_cuda(0, ctx->numNodesWithEdges, local_r, ctx)"]),
], host = True),
Kernel("BC_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("BC", ("ctx->gg", "__begin", "__end", "ctx->betweeness_centrality.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("BC_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["BC_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("BC_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["BC_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("BC_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["BC_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("Sanity_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('float &', 'DGAccumulator_sum'), ('float &', 'DGAccumulator_max'), ('float &', 'DGAccumulator_min'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<float>", "DGAccumulator_sumval", " = Shared<float>(1)")]),
CDecl([("HGAccumulator<float>", "_DGAccumulator_sum", "")]),
CBlock(["*(DGAccumulator_sumval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr()"]),
CDecl([("Shared<float>", "DGAccumulator_maxval", " = Shared<float>(1)")]),
CDecl([("HGReduceMax<float>", "_DGAccumulator_max", "")]),
CBlock(["*(DGAccumulator_maxval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_max.rv = DGAccumulator_maxval.gpu_wr_ptr()"]),
CDecl([("Shared<float>", "DGAccumulator_minval", " = Shared<float>(1)")]),
CDecl([("HGReduceMin<float>", "_DGAccumulator_min", "")]),
CBlock(["*(DGAccumulator_minval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_min.rv = DGAccumulator_minval.gpu_wr_ptr()"]),
Invoke("Sanity", ("ctx->gg", "__begin", "__end", "ctx->betweeness_centrality.data.gpu_wr_ptr()", "_DGAccumulator_sum", "_DGAccumulator_max", "_DGAccumulator_min")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr())"]),
CBlock(["DGAccumulator_max = *(DGAccumulator_maxval.cpu_rd_ptr())"]),
CBlock(["DGAccumulator_min = *(DGAccumulator_minval.cpu_rd_ptr())"]),
], host = True),
Kernel("Sanity_allNodes_cuda", [('float &', 'DGAccumulator_sum'), ('float &', 'DGAccumulator_max'), ('float &', 'DGAccumulator_min'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["Sanity_cuda(0, ctx->gg.nnodes, DGAccumulator_sum, DGAccumulator_max, DGAccumulator_min, ctx)"]),
], host = True),
Kernel("Sanity_masterNodes_cuda", [('float &', 'DGAccumulator_sum'), ('float &', 'DGAccumulator_max'), ('float &', 'DGAccumulator_min'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["Sanity_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_sum, DGAccumulator_max, DGAccumulator_min, ctx)"]),
], host = True),
Kernel("Sanity_nodesWithEdges_cuda", [('float &', 'DGAccumulator_sum'), ('float &', 'DGAccumulator_max'), ('float &', 'DGAccumulator_min'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["Sanity_cuda(0, ctx->numNodesWithEdges, DGAccumulator_sum, DGAccumulator_max, DGAccumulator_min, ctx)"]),
], host = True),
])
