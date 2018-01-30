from gg.ast import *
from gg.lib.graph import Graph
from gg.lib.wl import Worklist
from gg.ast.params import GraphParam
import cgen
G = Graph("graph")
WL = Worklist()
ast = Module([
CBlock([cgen.Include("kernels/reduce.cuh", system = False)], parse = False),
CBlock([cgen.Include("gen_cuda.cuh", system = False)], parse = False),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('uint32_t *', 'p_comp_old')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_comp_current[src] = graph.node_data[src]"]),
CBlock(["p_comp_old[src] = graph.node_data[src]"]),
]),
]),
]),
Kernel("FirstItr_ConnectedComp", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('uint32_t *', 'p_comp_old'), ('DynamicBitset&', 'bitset_comp_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_comp_old[src] = p_comp_current[src]"]),
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
Kernel("ConnectedComp", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('uint32_t *', 'p_comp_old'), ('DynamicBitset&', 'bitset_comp_current'), ('HGAccumulator<unsigned int>', 'DGAccumulator_accum')],
[
CDecl([("__shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage", "DGAccumulator_accum_ts", "")]),
CBlock(["DGAccumulator_accum.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_comp_old[src] > p_comp_current[src]",
[
CBlock(["p_comp_old[src] = p_comp_current[src]"]),
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
CBlock(["DGAccumulator_accum.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE>>(DGAccumulator_accum_ts)"], parse = False),
]),
Kernel("ConnectedCompSanityCheck", [G.param(), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('uint32_t *', 'p_comp_current'), ('HGAccumulator<uint64_t>', 'DGAccumulator_accum')],
[
CDecl([("__shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage", "DGAccumulator_accum_ts", "")]),
CBlock(["DGAccumulator_accum.thread_entry()"]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_comp_current[src] == graph.node_data[src]",
[
CBlock(["DGAccumulator_accum.reduce( 1)"]),
]),
]),
]),
CBlock(["DGAccumulator_accum.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE>>(DGAccumulator_accum_ts)"], parse = False),
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
Kernel("ConnectedComp_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<unsigned int>", "DGAccumulator_accumval", " = Shared<unsigned int>(1)")]),
CDecl([("HGAccumulator<unsigned int>", "_DGAccumulator_accum", "")]),
CBlock(["*(DGAccumulator_accumval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr()"]),
Invoke("ConnectedComp", ("ctx->gg", "__begin", "__end", "ctx->comp_current.data.gpu_wr_ptr()", "ctx->comp_old.data.gpu_wr_ptr()", "*(ctx->comp_current.is_updated.gpu_rd_ptr())", "_DGAccumulator_accum")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr())"]),
], host = True),
Kernel("ConnectedComp_allNodes_cuda", [('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedComp_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("ConnectedComp_masterNodes_cuda", [('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedComp_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("ConnectedComp_nodesWithEdges_cuda", [('unsigned int &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedComp_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("ConnectedCompSanityCheck_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('uint64_t &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CDecl([("Shared<uint64_t>", "DGAccumulator_accumval", " = Shared<uint64_t>(1)")]),
CDecl([("HGAccumulator<uint64_t>", "_DGAccumulator_accum", "")]),
CBlock(["*(DGAccumulator_accumval.cpu_wr_ptr()) = 0"]),
CBlock(["_DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr()"]),
Invoke("ConnectedCompSanityCheck", ("ctx->gg", "__begin", "__end", "ctx->comp_current.data.gpu_wr_ptr()", "_DGAccumulator_accum")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr())"]),
], host = True),
Kernel("ConnectedCompSanityCheck_allNodes_cuda", [('uint64_t &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedCompSanityCheck_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("ConnectedCompSanityCheck_masterNodes_cuda", [('uint64_t &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedCompSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, ctx)"]),
], host = True),
Kernel("ConnectedCompSanityCheck_nodesWithEdges_cuda", [('uint64_t &', 'DGAccumulator_accum'), ('struct CUDA_Context* ', 'ctx')],
[
CBlock(["ConnectedCompSanityCheck_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, ctx)"]),
], host = True),
])
