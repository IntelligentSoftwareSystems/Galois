from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("kcore_push_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph2", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_current_degree'), ('DynamicBitset&', 'bitset_current_degree')],
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
CDecl([("index_type", "dest_node", "")]),
CBlock(["dest_node = graph.getAbsDestination(current_edge)"]),
CBlock(["atomicTestAdd(&p_current_degree[dest_node], (uint32_t)1)"]),
CBlock(["bitset_current_degree.set(dest_node)"]),
]),
),
]),
]),
Kernel("InitializeGraph1", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_current_degree'), ('uint8_t *', 'p_flag'), ('uint32_t *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_flag[src]           = true"]),
CBlock(["p_trim[src]           = 0"]),
CBlock(["p_current_degree[src] = 0"]),
]),
]),
]),
Kernel("KCoreStep2", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_current_degree'), ('uint8_t *', 'p_flag'), ('uint32_t *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_flag[src]",
[
If("p_trim[src] > 0",
[
CBlock(["p_current_degree[src] = p_current_degree[src] - p_trim[src]"]),
]),
]),
CBlock(["p_trim[src] = 0"]),
]),
]),
]),
Kernel("KCoreStep1", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t', 'local_k_core_num'), ('uint32_t *', 'p_current_degree'), ('uint8_t *', 'p_flag'), ('uint32_t *', 'p_trim'), ('DynamicBitset&', 'bitset_trim'), ('HGAccumulator<unsigned int>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_flag[src]",
[
If("p_current_degree[src] < local_k_core_num",
[
CBlock(["p_flag[src] = false"]),
CBlock(["active_vertices.reduce( 1)"]),
], [ CBlock(["pop = false"]), ]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CBlock(["atomicTestAdd(&p_trim[dst], (uint32_t)1)"]),
CBlock(["bitset_trim.set(dst)"]),
]),
),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("KCoreSanityCheck", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint8_t *', 'p_flag'), ('HGAccumulator<uint64_t>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_flag[src]",
[
CBlock(["active_vertices.reduce( 1)"]),
]),
]),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("InitializeGraph2_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph2", ("ctx->gg", "__begin", "__end", "ctx->current_degree.data.gpu_wr_ptr()", "*(ctx->current_degree.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph2_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph2_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("InitializeGraph2_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph2_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("InitializeGraph2_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph2_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("InitializeGraph1_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph1", ("ctx->gg", "__begin", "__end", "ctx->current_degree.data.gpu_wr_ptr()", "ctx->flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph1_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph1_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("InitializeGraph1_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph1_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("InitializeGraph1_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph1_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("KCoreStep2_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("KCoreStep2", ("ctx->gg", "__begin", "__end", "ctx->current_degree.data.gpu_wr_ptr()", "ctx->flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("KCoreStep2_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreStep2_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("KCoreStep2_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreStep2_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("KCoreStep2_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreStep2_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("KCoreStep1_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('unsigned int &', 'active_vertices'), ('uint32_t', 'local_k_core_num'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<unsigned int>", "active_verticesval", " = Shared<unsigned int>(1)")]),
CDecl([("HGAccumulator<unsigned int>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("KCoreStep1", ("ctx->gg", "__begin", "__end", "local_k_core_num", "ctx->current_degree.data.gpu_wr_ptr()", "ctx->flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()", "*(ctx->trim.is_updated.gpu_rd_ptr())", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("KCoreStep1_allNodes_cuda", [('unsigned int &', 'active_vertices'), ('uint32_t', 'local_k_core_num'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreStep1_cuda(0, ctx->gg.nnodes, active_vertices, local_k_core_num, ctx)"]),
], host = True),
Kernel("KCoreStep1_masterNodes_cuda", [('unsigned int &', 'active_vertices'), ('uint32_t', 'local_k_core_num'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreStep1_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, local_k_core_num, ctx)"]),
], host = True),
Kernel("KCoreStep1_nodesWithEdges_cuda", [('unsigned int &', 'active_vertices'), ('uint32_t', 'local_k_core_num'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreStep1_cuda(0, ctx->numNodesWithEdges, active_vertices, local_k_core_num, ctx)"]),
], host = True),
Kernel("KCoreSanityCheck_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint64_t>", "active_verticesval", " = Shared<uint64_t>(1)")]),
CDecl([("HGAccumulator<uint64_t>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("KCoreSanityCheck", ("ctx->gg", "__begin", "__end", "ctx->flag.data.gpu_wr_ptr()", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("KCoreSanityCheck_allNodes_cuda", [('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreSanityCheck_cuda(0, ctx->gg.nnodes, active_vertices, ctx)"]),
], host = True),
Kernel("KCoreSanityCheck_masterNodes_cuda", [('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx)"]),
], host = True),
Kernel("KCoreSanityCheck_nodesWithEdges_cuda", [('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["KCoreSanityCheck_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx)"]),
], host = True),
])
