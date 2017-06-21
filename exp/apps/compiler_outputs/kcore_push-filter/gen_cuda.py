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
CDeclGlobal([("unsigned int *", "P_CURRENT_DEGREE", "")]),
CDeclGlobal([("bool *", "P_FLAG", "")]),
CDeclGlobal([("unsigned int *", "P_TRIM", "")]),
Kernel("InitializeGraph2", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_degree')],
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
CBlock(["atomicAdd(&p_current_degree[dest_node], (unsigned int)1)"]),
]),
),
]),
]),
Kernel("InitializeGraph1", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_degree'), ('bool *', 'p_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_flag[src] = true"]),
CBlock(["p_trim[src] = 0"]),
CBlock(["p_current_degree[src] = 0"]),
]),
]),
]),
Kernel("KCoreStep2", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_current_degree'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_trim[src] > 0",
[
CBlock(["p_current_degree[src] -= p_trim[src]"]),
CBlock(["p_trim[src] = 0"]),
]),
]),
]),
]),
Kernel("KCoreStep1", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int', 'local_k_core_num'), ('unsigned int *', 'p_current_degree'), ('bool *', 'p_flag'), ('unsigned int *', 'p_trim')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
If("p_flag[src]",
[
If("p_current_degree[src] < local_k_core_num",
[
CBlock(["p_flag[src] = false"]),
ReturnFromParallelFor(" 1"),
], [ CBlock(["pop = false"]), ]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("current_edge", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(current_edge)"]),
CBlock(["atomicAdd(&p_trim[dst], (unsigned int)1)"]),
]),
),
]),
]),
Kernel("InitializeGraph2_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph2", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_degree.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph2_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph2_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("InitializeGraph1_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph1", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_degree.data.gpu_wr_ptr()", "ctx->flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph1_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph1_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("KCoreStep2_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("KCoreStep2", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->current_degree.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("KCoreStep2_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["KCoreStep2_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("KCoreStep1_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('unsigned int', 'local_k_core_num'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("KCoreStep1", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_k_core_num", "ctx->current_degree.data.gpu_wr_ptr()", "ctx->flag.data.gpu_wr_ptr()", "ctx->trim.data.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("KCoreStep1_all_cuda", [('int &', '__retval'), ('unsigned int', 'local_k_core_num'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["KCoreStep1_cuda(0, ctx->nowned, __retval, local_k_core_num, ctx)"]),
], host = True),
])
