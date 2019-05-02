from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("sssp_pull_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('uint32_t *', 'p_dist_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_dist_current[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity"]),
]),
]),
]),
Kernel("SSSP", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_dist_current'), ('DynamicBitset&', 'bitset_dist_current'), ('HGAccumulator<unsigned int>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = p_dist_current[dst] + graph.getAbsWeight(jj)"]),
CDecl([("uint32_t", "old_dist", "")]),
CBlock(["old_dist = atomicTestMin(&p_dist_current[src], new_dist)"]),
If("old_dist > new_dist",
[
CBlock(["bitset_dist_current.set(src)"]),
CBlock(["active_vertices.reduce( 1)"]),
]),
]),
),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("SSSPSanityCheck", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const uint32_t ', 'local_infinity'), ('uint32_t *', 'p_dist_current'), ('HGAccumulator<uint64_t>', 'DGAccumulator_sum'), ('HGAccumulator<uint64_t>', 'dg_avg'), ('HGReduceMax<uint32_t>', 'DGMax')],
[
CDecl([("__shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage", "DGAccumulator_sum_ts", "")]),
CBlock(["DGAccumulator_sum.thread_entry()"]),
CDecl([("__shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage", "dg_avg_ts", "")]),
CBlock(["dg_avg.thread_entry()"]),
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
CBlock(["dg_avg.reduce( p_dist_current[src])"]),
]),
]),
]),
CBlock(["DGAccumulator_sum.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(DGAccumulator_sum_ts)"], parse = False),
CBlock(["dg_avg.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(dg_avg_ts)"], parse = False),
CBlock(["DGMax.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(DGMax_ts)"], parse = False),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const uint32_t &', 'local_infinity'), ('unsigned long long', 'local_src_node'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "__begin", "__end", "local_infinity", "local_src_node", "ctx->dist_current.data.gpu_wr_ptr()")),
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
Kernel("SSSP_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<unsigned int>", "active_verticesval", " = Shared<unsigned int>(1)")]),
CDecl([("HGAccumulator<unsigned int>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("SSSP", ("ctx->gg", "__begin", "__end", "ctx->dist_current.data.gpu_wr_ptr()", "*(ctx->dist_current.is_updated.gpu_rd_ptr())", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("SSSP_allNodes_cuda", [('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->gg.nnodes, active_vertices, ctx)"]),
], host = True),
Kernel("SSSP_masterNodes_cuda", [('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx)"]),
], host = True),
Kernel("SSSP_nodesWithEdges_cuda", [('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSP_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx)"]),
], host = True),
Kernel("SSSPSanityCheck_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint64_t &', 'DGAccumulator_sum'), ('uint64_t &', 'dg_avg'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint64_t>", "DGAccumulator_sumval", " = Shared<uint64_t>(1)")]),
CDecl([("HGAccumulator<uint64_t>", "_DGAccumulator_sum", "")]),
CBlock(["*(DGAccumulator_sumval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr()"]),
CDecl([("Shared<uint64_t>", "dg_avgval", " = Shared<uint64_t>(1)")]),
CDecl([("HGAccumulator<uint64_t>", "_dg_avg", "")]),
CBlock(["*(dg_avgval.cpu_wr_ptr()) = 0"]),
CBlock(["_dg_avg.rv = dg_avgval.gpu_wr_ptr()"]),
CDecl([("Shared<uint32_t>", "DGMaxval", " = Shared<uint32_t>(1)")]),
CDecl([("HGReduceMax<uint32_t>", "_DGMax", "")]),
CBlock(["*(DGMaxval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGMax.rv = DGMaxval.gpu_wr_ptr()"]),
Invoke("SSSPSanityCheck", ("ctx->gg", "__begin", "__end", "local_infinity", "ctx->dist_current.data.gpu_wr_ptr()", "_DGAccumulator_sum", "_dg_avg", "_DGMax")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr())"]),
CBlock(["dg_avg = *(dg_avgval.cpu_rd_ptr())"]),
CBlock(["DGMax = *(DGMaxval.cpu_rd_ptr())"]),
], host = True),
Kernel("SSSPSanityCheck_allNodes_cuda", [('uint64_t &', 'DGAccumulator_sum'), ('uint64_t &', 'dg_avg'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSPSanityCheck_cuda(0, ctx->gg.nnodes, DGAccumulator_sum, dg_avg, DGMax, local_infinity, ctx)"]),
], host = True),
Kernel("SSSPSanityCheck_masterNodes_cuda", [('uint64_t &', 'DGAccumulator_sum'), ('uint64_t &', 'dg_avg'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSPSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_sum, dg_avg, DGMax, local_infinity, ctx)"]),
], host = True),
Kernel("SSSPSanityCheck_nodesWithEdges_cuda", [('uint64_t &', 'DGAccumulator_sum'), ('uint64_t &', 'dg_avg'), ('uint32_t &', 'DGMax'), ('const uint32_t &', 'local_infinity'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["SSSPSanityCheck_cuda(0, ctx->numNodesWithEdges, DGAccumulator_sum, dg_avg, DGMax, local_infinity, ctx)"]),
], host = True),
])
