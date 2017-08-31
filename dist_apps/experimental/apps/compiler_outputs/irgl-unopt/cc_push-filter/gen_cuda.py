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
CDeclGlobal([("unsigned int *", "P_COMP_CURRENT", "")]),
CDeclGlobal([("unsigned int *", "P_COMP_OLD", "")]),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_comp_current'), ('unsigned int *', 'p_comp_old')],
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
Kernel("FirstItr_ConnectedComp", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_comp_current'), ('unsigned int *', 'p_comp_old')],
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
CDecl([("unsigned int", "new_dist", "")]),
CBlock(["new_dist = p_comp_current[src]"]),
CBlock(["atomicMin(&p_comp_current[dst], new_dist)"]),
]),
),
]),
]),
Kernel("ConnectedComp", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_comp_current'), ('unsigned int *', 'p_comp_old')],
[
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
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("unsigned int", "new_dist", "")]),
CBlock(["new_dist = p_comp_current[src]"]),
CBlock(["atomicMin(&p_comp_current[dst], new_dist)"]),
]),
),
ReturnFromParallelFor(" 1"),
]),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->comp_current.gpu_wr_ptr()", "ctx->comp_old.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("FirstItr_ConnectedComp_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("FirstItr_ConnectedComp", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->comp_current.gpu_wr_ptr()", "ctx->comp_old.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("FirstItr_ConnectedComp_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["FirstItr_ConnectedComp_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("ConnectedComp_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("ConnectedComp", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->comp_current.gpu_wr_ptr()", "ctx->comp_old.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("ConnectedComp_all_cuda", [('int &', '__retval'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["ConnectedComp_cuda(0, ctx->nowned, __retval, ctx)"]),
], host = True),
])
