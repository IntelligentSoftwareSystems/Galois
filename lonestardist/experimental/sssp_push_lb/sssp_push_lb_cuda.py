from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("sssp_push_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('uint32_t *', 'p_dist_current'), ('uint32_t *', 'p_dist_old')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_dist_current[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity"]),
CBlock(["p_dist_old[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity"]),
]),
]),
]),
Kernel("FirstItr_SSSP", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_dist_current'), ('uint32_t *', 'p_dist_old'), ('DynamicBitset&', 'bitset_dist_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_dist_old[src] = p_dist_current[src]"]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = graph.getAbsWeight(jj) + p_dist_current[src]"]),
CDecl([("uint32_t", "old_dist", "")]),
CBlock(["old_dist = atomicTestMin(&p_dist_current[dst], new_dist)"]),
If("old_dist > new_dist",
[
CBlock(["bitset_dist_current.set(dst)"]),
]),
]),
),
]),
]),
Kernel("SSSP", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_dist_current'), ('uint32_t *', 'p_dist_old'), ('DynamicBitset&', 'bitset_dist_current'), ('HGAccumulator<unsigned int>', 'DGAccumulator_accum')],
[
CDecl([("__shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage", "DGAccumulator_accum_ts", "")]),
CBlock(["DGAccumulator_accum.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_dist_old[src] > p_dist_current[src]",
[
CBlock(["p_dist_old[src] = p_dist_current[src]"]),
CBlock(["DGAccumulator_accum.reduce( 1)"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = graph.getAbsWeight(jj) + p_dist_current[src]"]),
CDecl([("uint32_t", "old_dist", "")]),
CBlock(["old_dist = atomicTestMin(&p_dist_current[dst], new_dist)"]),
If("old_dist > new_dist",
[
CBlock(["bitset_dist_current.set(dst)"]),
]),
]),
),
]),
CBlock(["DGAccumulator_accum.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE>>(DGAccumulator_accum_ts)"], parse = False),
]),
Kernel("SSSPSanityCheck", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_dist_current'), ('HGAccumulator<uint64_t>', 'DGAccumulator_sum'), ('HGReduceMax<uint32_t>', 'DGMax')],
[
CDecl([("__shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage", "DGAccumulator_sum_ts", "")]),
CBlock(["DGAccumulator_sum.thread_entry()"]),
CDecl([("__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage", "DGMax_ts", "")]),
CBlock(["DGMax.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_dist_current[src] < local_infinity",
[
CBlock(["DGAccumulator_sum.reduce( 1)"]),
CBlock(["DGMax.reduce(p_dist_current[src])"]),
]),
]),
]),
CBlock(["DGAccumulator_sum.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE>>(DGAccumulator_sum_ts)"], parse = False),
CBlock(["DGMax.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(DGMax_ts)"], parse = False),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "__begin", "__end", "local_infinity", "local_src_node", "ctx->dist_current.data.gpu_wr_ptr()", "ctx->dist_old.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_allNodes_cuda", [('const uint32_t &', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->gg.nnodes, local_infinity, local_src_node, ctx)"]),
], host = True),
Kernel("InitializeGraph_masterNodes_cuda", [('const uint32_t &', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, local_src_node, ctx)"]),
], host = True),
Kernel("InitializeGraph_nodesWithEdges_cuda", [('const uint32_t &', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->numNodesWithEdges, local_infinity, local_src_node, ctx)"]),
], host = True),
Kernel("FirstItr_SSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("FirstItr_SSSP", ("ctx->gg", "__begin", "__end", "ctx->dist_current.data.gpu_wr_ptr()", "ctx->dist_old.data.gpu_wr_ptr()", "*(ctx->dist_current.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("FirstItr_SSSP_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstItr_SSSP_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("FirstItr_SSSP_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstItr_SSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("FirstItr_SSSP_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstItr_SSSP_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("SSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<unsigned int>", "DGAccumulator_accumval", " = Shared<unsigned int>(1)")]),
CDecl([("HGAccumulator<unsigned int>", "_DGAccumulator_accum", "")]),
CBlock(["*(DGAccumulator_accumval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr()"]),
Invoke("SSSP", ("ctx->gg", "__begin", "__end", "ctx->dist_current.data.gpu_wr_ptr()", "ctx->dist_old.data.gpu_wr_ptr()", "*(ctx->dist_current.is_updated.gpu_rd_ptr())", "_DGAccumulator_accum")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr())"]),
], host = True),
Kernel("SSSP_allNodes_cuda", [('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("SSSP_masterNodes_cuda", [('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("SSSP_nodesWithEdges_cuda", [('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("SSSPSanityCheck_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint64_t &', 'DGAccumulator_sum'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint64_t>", "DGAccumulator_sumval", " = Shared<uint64_t>(1)")]),
CDecl([("HGAccumulator<uint64_t>", "_DGAccumulator_sum", "")]),
CBlock(["*(DGAccumulator_sumval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr()"]),
CDecl([("Shared<uint32_t>", "DGMaxval", " = Shared<uint32_t>(1)")]),
CDecl([("HGReduceMax<uint32_t>", "_DGMax", "")]),
CBlock(["*(DGMaxval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGMax.rv = DGMaxval.gpu_wr_ptr()"]),
Invoke("SSSPSanityCheck", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->dist_current.data.gpu_wr_ptr()", "_DGAccumulator_sum", "_DGMax")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr())"]),
CBlock(["DGMax = *(DGMaxval.cpu_rd_ptr())"]),
], host = True),
Kernel("SSSPSanityCheck_allNodes_cuda", [('uint64_t &', 'DGAccumulator_sum'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSPSanityCheck_cuda(0, ctx->gg.nnodes, DGAccumulator_sum, DGMax, local_infinity, ctx)"]),
], host = True),
Kernel("SSSPSanityCheck_masterNodes_cuda", [('uint64_t &', 'DGAccumulator_sum'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSPSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_sum, DGMax, local_infinity, ctx)"]),
], host = True),
Kernel("SSSPSanityCheck_nodesWithEdges_cuda", [('uint64_t &', 'DGAccumulator_sum'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSPSanityCheck_cuda(0, ctx->numNodesWithEdges, DGAccumulator_sum, DGMax, local_infinity, ctx)"]),
], host = True),
])
