from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("cc_push_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('uint32_t *', 'p_comp_old')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_comp_current[src] = graph.node_data[src]"]),
CBlock(["p_comp_old[src]     = graph.node_data[src]"]),
]),
]),
]),
Kernel("FirstItr_ConnectedComp", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('uint32_t *', 'p_comp_old'), ('DynamicBitset&', 'bitset_comp_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_comp_old[src]  = p_comp_current[src]"]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = p_comp_current[src]"]),
CDecl([("uint32_t", "old_dist", "")]),
CBlock(["old_dist = atomicTestMin(&p_comp_current[dst], new_dist)"]),
If("old_dist > new_dist",
[
CBlock(["bitset_comp_current.set(dst)"]),
]),
]),
),
]),
]),
Kernel("ConnectedComp", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('uint32_t *', 'p_comp_old'), ('DynamicBitset&', 'bitset_comp_current'), ('HGAccumulator<unsigned int>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_comp_old[src] > p_comp_current[src]",
[
CBlock(["p_comp_old[src] = p_comp_current[src]"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CBlock(["active_vertices.reduce( 1)"]),
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("uint32_t", "new_dist", "")]),
CBlock(["new_dist = p_comp_current[src]"]),
CDecl([("uint32_t", "old_dist", "")]),
CBlock(["old_dist = atomicTestMin(&p_comp_current[dst], new_dist)"]),
If("old_dist > new_dist",
[
CBlock(["bitset_comp_current.set(dst)"]),
]),
]),
),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("ConnectedCompSanityCheck", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('HGAccumulator<uint64_t>', 'active_vertices')],
[
CDecl([("__shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage", "active_vertices_ts", "")]),
CBlock(["active_vertices.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_comp_current[src] == graph.node_data[src]",
[
CBlock(["active_vertices.reduce( 1)"]),
]),
]),
]),
CBlock(["active_vertices.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(active_vertices_ts)"], parse = False),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "__begin", "__end", "ctx->comp_current.data.gpu_wr_ptr()", "ctx->comp_old.data.gpu_wr_ptr()")),
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
Kernel("FirstItr_ConnectedComp_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("FirstItr_ConnectedComp", ("ctx->gg", "__begin", "__end", "ctx->comp_current.data.gpu_wr_ptr()", "ctx->comp_old.data.gpu_wr_ptr()", "*(ctx->comp_current.is_updated.gpu_rd_ptr())")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("FirstItr_ConnectedComp_allNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstItr_ConnectedComp_cuda(0, ctx->gg.nnodes, ctx)"]),
], host = True),
Kernel("FirstItr_ConnectedComp_masterNodes_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstItr_ConnectedComp_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx)"]),
], host = True),
Kernel("FirstItr_ConnectedComp_nodesWithEdges_cuda", [('struct CUDA_Context* ', 'ctx')],
[
CBlock(["FirstItr_ConnectedComp_cuda(0, ctx->numNodesWithEdges, ctx)"]),
], host = True),
Kernel("ConnectedComp_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<unsigned int>", "active_verticesval", " = Shared<unsigned int>(1)")]),
CDecl([("HGAccumulator<unsigned int>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("ConnectedComp", ("ctx->gg", "__begin", "__end", "ctx->comp_current.data.gpu_wr_ptr()", "ctx->comp_old.data.gpu_wr_ptr()", "*(ctx->comp_current.is_updated.gpu_rd_ptr())", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("ConnectedComp_allNodes_cuda", [('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedComp_cuda(0, ctx->gg.nnodes, active_vertices, ctx)"]),
], host = True),
Kernel("ConnectedComp_masterNodes_cuda", [('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedComp_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx)"]),
], host = True),
Kernel("ConnectedComp_nodesWithEdges_cuda", [('unsigned int &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedComp_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx)"]),
], host = True),
Kernel("ConnectedCompSanityCheck_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint64_t>", "active_verticesval", " = Shared<uint64_t>(1)")]),
CDecl([("HGAccumulator<uint64_t>", "_active_vertices", "")]),
CBlock(["*(active_verticesval.cpu_wr_ptr()) = 0"]),
CBlock(["_active_vertices.rv = active_verticesval.gpu_wr_ptr()"]),
Invoke("ConnectedCompSanityCheck", ("ctx->gg", "__begin", "__end", "ctx->comp_current.data.gpu_wr_ptr()", "_active_vertices")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["active_vertices = *(active_verticesval.cpu_rd_ptr())"]),
], host = True),
Kernel("ConnectedCompSanityCheck_allNodes_cuda", [('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedCompSanityCheck_cuda(0, ctx->gg.nnodes, active_vertices, ctx)"]),
], host = True),
Kernel("ConnectedCompSanityCheck_masterNodes_cuda", [('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedCompSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx)"]),
], host = True),
Kernel("ConnectedCompSanityCheck_nodesWithEdges_cuda", [('uint64_t &', 'active_vertices'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedCompSanityCheck_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx)"]),
], host = True),
])
