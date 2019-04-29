from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("bc_push_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality'), ('float *', 'p_dependency'), ('uint32_t *', 'p_num_predecessors'), ('ShortPathType *', 'p_num_shortest_paths'), ('uint32_t *', 'p_num_successors'), ('uint8_t *', 'p_propagation_flag'), ('ShortPathType *', 'p_to_add'), ('float *', 'p_to_add_float'), ('uint32_t *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_betweeness_centrality[src] = 0"]),
CBlock(["p_num_shortest_paths[src]    = 0"]),
CBlock(["p_num_successors[src]        = 0"]),
CBlock(["p_num_predecessors[src]      = 0"]),
CBlock(["p_trim[src]                  = 0"]),
CBlock(["p_to_add[src]                = 0"]),
CBlock(["p_to_add_float[src]          = 0"]),
CBlock(["p_dependency[src]            = 0"]),
CBlock(["p_propagation_flag[src]      = false"]),
]),
]),
]),
Kernel("InitializeIteration", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint64_t ', 'local_current_src_node'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('float *', 'p_dependency'), ('uint32_t *', 'p_num_predecessors'), ('ShortPathType *', 'p_num_shortest_paths'), ('uint32_t *', 'p_num_successors'), ('uint32_t *', 'p_old_length'), ('uint8_t *', 'p_propagation_flag')],
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
CBlock(["p_old_length[src]         = local_infinity"]),
CBlock(["p_num_shortest_paths[src] = 0"]),
CBlock(["p_propagation_flag[src]   = false"]),
],
[
CBlock(["p_current_length[src]     = 0"]),
CBlock(["p_old_length[src]         = 0"]),
CBlock(["p_num_shortest_paths[src] = 1"]),
CBlock(["p_propagation_flag[src]   = true"]),
]),
CBlock(["p_num_predecessors[src] = 0"]),
CBlock(["p_num_successors[src]   = 0"]),
CBlock(["p_dependency[src]       = 0"]),
]),
]),
]),
Kernel("FirstIterationSSSP", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_current_length'), ('DynamicBitset&', 'bitset_current_length')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = edge_weight + p_current_length[src]"]),
CBlock(["atomicTestMin(&p_current_length[dst], new_dist)"]),
CBlock(["bitset_current_length.set(dst)"]),
]),
),
]),
]),
Kernel("SSSP", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_current_length'), ('uint32_t *', 'p_old_length'), ('DynamicBitset&', 'bitset_current_length'), ('HGAccumulator<uint32_t>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_old_length[src] > p_current_length[src]",
[
CBlock(["p_old_length[src] = p_current_length[src]"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = edge_weight + p_current_length[src]"]),
CDecl([("uint32_t", "old", "")]),
CBlock(["old = atomicTestMin(&p_current_length[dst], new_dist)"]),
If("old > new_dist",
[
CBlock(["bitset_current_length.set(dst)"]),
CBlock(["active_vertices.reduce( 1)"]),
]),
]),
),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("PredAndSucc", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('uint32_t *', 'p_num_predecessors'), ('uint32_t *', 'p_num_successors'), ('DynamicBitset&', 'bitset_num_predecessors'), ('DynamicBitset&', 'bitset_num_successors')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
If("(p_current_length[src] + edge_weight) == p_current_length[dst]",
[
CBlock(["atomicTestAdd(&p_num_successors[src], (unsigned int)1)"]),
CBlock(["atomicTestAdd(&p_num_predecessors[dst], (unsigned int)1)"]),
CBlock(["bitset_num_successors.set(src)"]),
CBlock(["bitset_num_predecessors.set(dst)"]),
]),
]),
),
]),
]),
Kernel("NumShortestPathsChanges", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('uint32_t *', 'p_num_predecessors'), ('ShortPathType *', 'p_num_shortest_paths'), ('uint8_t *', 'p_propagation_flag'), ('ShortPathType *', 'p_to_add'), ('uint32_t *', 'p_trim'), ('DynamicBitset&', 'bitset_num_shortest_paths')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
If("p_trim[src] > 0",
[
CBlock(["p_num_predecessors[src] = p_num_predecessors[src] - p_trim[src]"]),
CBlock(["p_trim[src]             = 0"]),
If("p_num_predecessors[src] == 0",
[
CBlock(["p_propagation_flag[src] = true"]),
]),
]),
If("p_to_add[src] > 0",
[
CBlock(["p_num_shortest_paths[src] += p_to_add[src]"]),
CBlock(["p_to_add[src] = 0"]),
CBlock(["bitset_num_shortest_paths.set(src)"]),
]),
]),
]),
]),
]),
Kernel("NumShortestPaths", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('ShortPathType *', 'p_num_shortest_paths'), ('uint8_t *', 'p_propagation_flag'), ('ShortPathType *', 'p_to_add'), ('uint32_t *', 'p_trim'), ('DynamicBitset&', 'bitset_to_add'), ('DynamicBitset&', 'bitset_trim'), ('HGAccumulator<uint32_t>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
If("p_propagation_flag[src]",
[
CBlock(["p_propagation_flag[src] = false"]),
], [ CBlock(["pop = false"]), ]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
CDecl([("ShortPathType", "paths_to_add", "")]),
CBlock(["paths_to_add = p_num_shortest_paths[src]"]),
If("(p_current_length[src] + edge_weight) == p_current_length[dst]",
[
CBlock(["atomicTestAdd(&p_to_add[dst], paths_to_add)"]),
CBlock(["atomicTestAdd(&p_trim[dst], (unsigned int)1)"]),
CBlock(["bitset_to_add.set(dst)"]),
CBlock(["bitset_trim.set(dst)"]),
CBlock(["active_vertices.reduce( 1)"]),
]),
]),
),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("PropagationFlagUpdate", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('uint32_t *', 'p_num_successors'), ('uint8_t *', 'p_propagation_flag'), ('DynamicBitset&', 'bitset_propagation_flag')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
If("p_num_successors[src] == 0",
[
CBlock(["p_propagation_flag[src] = true"]),
CBlock(["bitset_propagation_flag.set(src)"]),
],
[
]),
]),
]),
]),
]),
Kernel("DependencyPropChanges", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('float *', 'p_dependency'), ('uint32_t *', 'p_num_successors'), ('uint8_t *', 'p_propagation_flag'), ('float *', 'p_to_add_float'), ('uint32_t *', 'p_trim'), ('DynamicBitset&', 'bitset_dependency'), ('DynamicBitset&', 'bitset_propagation_flag')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
If("p_to_add_float[src] > 0.0",
[
CBlock(["p_dependency[src] += p_to_add_float[src]"]),
CBlock(["p_to_add_float[src] = 0.0"]),
CBlock(["bitset_dependency.set(src)"]),
]),
If("(p_num_successors[src] == 0) && p_propagation_flag[src]",
[
CBlock(["p_propagation_flag[src] = false"]),
CBlock(["bitset_propagation_flag.set(src)"]),
],
[
If("p_trim[src] > 0",
[
CBlock(["p_num_successors[src] = p_num_successors[src] - p_trim[src]"]),
CBlock(["p_trim[src]           = 0"]),
If("p_num_successors[src] == 0",
[
CBlock(["p_propagation_flag[src] = true"]),
CBlock(["bitset_propagation_flag.set(src)"]),
]),
]),
]),
]),
]),
]),
]),
Kernel("DependencyPropagation", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint64_t ', 'local_current_src_node'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_current_length'), ('float *', 'p_dependency'), ('ShortPathType *', 'p_num_shortest_paths'), ('uint32_t *', 'p_num_successors'), ('uint8_t *', 'p_propagation_flag'), ('float *', 'p_to_add_float'), ('uint32_t *', 'p_trim'), ('DynamicBitset&', 'bitset_to_add_float'), ('DynamicBitset&', 'bitset_trim'), ('HGAccumulator<uint32_t>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_current_length[src] != local_infinity",
[
If("p_num_successors[src] > 0",
[
If("graph.node_data[src] == local_current_src_node",
[
CBlock(["p_num_successors[src] = 0"]),
]),
If("graph.node_data[src] != local_current_src_node",
[
], [ CBlock(["pop = false"]), ]),
], [ CBlock(["pop = false"]), ]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CDecl([("int", "edge_weight", "")]),
CBlock(["edge_weight = 1"]),
If("p_propagation_flag[dst]",
[
If("(p_current_length[src] + edge_weight) == p_current_length[dst]",
[
CBlock(["atomicTestAdd(&p_trim[src], (unsigned int)1)"]),
CDecl([("float", "contrib", "")]),
CBlock(["contrib = p_num_shortest_paths[src]"]),
CBlock(["contrib /= p_num_shortest_paths[dst]"]),
CBlock(["contrib *= (1.0 + p_dependency[dst])"]),
CBlock(["atomicTestAdd(&p_to_add_float[src], contrib)"]),
CBlock(["bitset_trim.set(src)"]),
CBlock(["bitset_to_add_float.set(src)"]),
CBlock(["active_vertices.reduce( 1)"]),
]),
]),
]),
),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("BC", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('float *', 'p_betweeness_centrality'), ('float *', 'p_dependency')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_dependency[src] > 0",
[
CBlock(["atomicTestAdd(&p_betweeness_centrality[src], p_dependency[src])"]),
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
Invoke("InitializeGraph", ("ctx->gg", "__begin", "__end", "ctx->betweeness_centrality.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()", "ctx->to_add.data.gpu_wr_ptr()", "ctx->to_add_float.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
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
Invoke("InitializeIteration", ("ctx->gg", "__begin", "__end", "local_current_src_node", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->old_length.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()")),
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
Kernel("FirstIterationSSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("FirstIterationSSSP", ("ctx->gg", "__begin", "__end", "ctx->current_length.data.gpu_wr_ptr()", "*(ctx->current_length.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("FirstIterationSSSP_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstIterationSSSP_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("FirstIterationSSSP_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstIterationSSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("FirstIterationSSSP_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstIterationSSSP_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("SSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint32_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint32_t>", "active_verticesval", " = Shared<uint32_t>(1)")]),
CDecl([("HGAccumulator<uint32_t>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("SSSP", ("ctx->gg", "__begin", "__end", "ctx->current_length.data.gpu_wr_ptr()", "ctx->old_length.data.gpu_wr_ptr()", "*(ctx->current_length.is_updated.gpu_rd_ptr())", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("SSSP_allNodes_cuda", [('uint32_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->gg.nnodes, active_vertices, ctx)"]),
], host = True),
Kernel("SSSP_masterNodes_cuda", [('uint32_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx)"]),
], host = True),
Kernel("SSSP_nodesWithEdges_cuda", [('uint32_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx)"]),
], host = True),
Kernel("PredAndSucc_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("PredAndSucc", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "*(ctx->num_predecessors.is_updated.gpu_rd_ptr())", "*(ctx->num_successors.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PredAndSucc_allNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["PredAndSucc_cuda(0, ctx->gg.nnodes, local_infinity, ctx)"]),
], host = True),
Kernel("PredAndSucc_masterNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["PredAndSucc_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx)"]),
], host = True),
Kernel("PredAndSucc_nodesWithEdges_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["PredAndSucc_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx)"]),
], host = True),
Kernel("NumShortestPathsChanges_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("NumShortestPathsChanges", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_predecessors.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()", "ctx->to_add.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()", "*(ctx->num_shortest_paths.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("NumShortestPathsChanges_allNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["NumShortestPathsChanges_cuda(0, ctx->gg.nnodes, local_infinity, ctx)"]),
], host = True),
Kernel("NumShortestPathsChanges_masterNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["NumShortestPathsChanges_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx)"]),
], host = True),
Kernel("NumShortestPathsChanges_nodesWithEdges_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["NumShortestPathsChanges_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx)"]),
], host = True),
Kernel("NumShortestPaths_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint32_t>", "active_verticesval", " = Shared<uint32_t>(1)")]),
CDecl([("HGAccumulator<uint32_t>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("NumShortestPaths", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()", "ctx->to_add.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()", "*(ctx->to_add.is_updated.gpu_rd_ptr())", "*(ctx->trim.is_updated.gpu_rd_ptr())", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("NumShortestPaths_allNodes_cuda", [('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["NumShortestPaths_cuda(0, ctx->gg.nnodes, active_vertices, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("NumShortestPaths_masterNodes_cuda", [('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["NumShortestPaths_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("NumShortestPaths_nodesWithEdges_cuda", [('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["NumShortestPaths_cuda(0, ctx->numNodesWithEdges, active_vertices, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("PropagationFlagUpdate_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("PropagationFlagUpdate", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()", "*(ctx->propagation_flag.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PropagationFlagUpdate_allNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["PropagationFlagUpdate_cuda(0, ctx->gg.nnodes, local_infinity, ctx)"]),
], host = True),
Kernel("PropagationFlagUpdate_masterNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["PropagationFlagUpdate_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx)"]),
], host = True),
Kernel("PropagationFlagUpdate_nodesWithEdges_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["PropagationFlagUpdate_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx)"]),
], host = True),
Kernel("DependencyPropChanges_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("DependencyPropChanges", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()", "ctx->to_add_float.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()", "*(ctx->dependency.is_updated.gpu_rd_ptr())", "*(ctx->propagation_flag.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("DependencyPropChanges_allNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["DependencyPropChanges_cuda(0, ctx->gg.nnodes, local_infinity, ctx)"]),
], host = True),
Kernel("DependencyPropChanges_masterNodes_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["DependencyPropChanges_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx)"]),
], host = True),
Kernel("DependencyPropChanges_nodesWithEdges_cuda", [('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["DependencyPropChanges_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx)"]),
], host = True),
Kernel("DependencyPropagation_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint32_t>", "active_verticesval", " = Shared<uint32_t>(1)")]),
CDecl([("HGAccumulator<uint32_t>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("DependencyPropagation", ("ctx->gg", "__begin", "__end", "local_current_src_node", "local_infinity", "ctx->current_length.data.gpu_wr_ptr()", "ctx->dependency.data.gpu_wr_ptr()", "ctx->num_shortest_paths.data.gpu_wr_ptr()", "ctx->num_successors.data.gpu_wr_ptr()", "ctx->propagation_flag.data.gpu_wr_ptr()", "ctx->to_add_float.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()", "*(ctx->to_add_float.is_updated.gpu_rd_ptr())", "*(ctx->trim.is_updated.gpu_rd_ptr())", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("DependencyPropagation_allNodes_cuda", [('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["DependencyPropagation_cuda(0, ctx->gg.nnodes, active_vertices, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("DependencyPropagation_masterNodes_cuda", [('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["DependencyPropagation_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, local_infinity, local_current_src_node, ctx)"]),
], host = True),
Kernel("DependencyPropagation_nodesWithEdges_cuda", [('uint32_t &', 'active_vertices'), ('const uint32_t &', 'local_infinity'), ('const uint64_t &', 'local_current_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["DependencyPropagation_cuda(0, ctx->numNodesWithEdges, active_vertices, local_infinity, local_current_src_node, ctx)"]),
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
