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
CDeclGlobal([("int *", "P_NOUT", "")]),
CDeclGlobal([("float *", "P_VALUE", "")]),
Kernel("ResetGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('int *', 'p_nout'), ('float *', 'p_value')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_value[src] = 0"]),
CBlock(["p_nout[src] = 0"]),
]),
]),
]),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const float ', 'local_alpha'), ('int *', 'p_nout'), ('float *', 'p_value')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_value[src] = local_alpha"]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CBlock(["atomicAdd(&p_nout[dst], 1)"]),
]),
),
]),
]),
Kernel("PageRank", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const float ', 'local_alpha'), ('float', 'local_tolerance'), ('int *', 'p_nout'), ('float *', 'p_value')],
[
CDecl([("float", "sum", "")]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["sum = 0"]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CDecl([("unsigned int", "dnout", "")]),
CBlock(["dnout = p_nout[dst]"]),
If("dnout > 0",
[
CBlock(["sum += p_value[dst]/dnout"]),
]),
]),
),
CDecl([("float", "pr_value", "")]),
CBlock(["pr_value = sum*(1.0 - local_alpha) + local_alpha"]),
CDecl([("float", "diff", "")]),
CBlock(["diff = fabs(pr_value - p_value[src])"]),
If("diff > local_tolerance",
[
CBlock(["p_value[src] = pr_value"]),
ReturnFromParallelFor(" 1"),
]),
]),
]),
Kernel("ResetGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("ResetGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->nout.data.gpu_wr_ptr()", "ctx->value.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("ResetGraph_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["ResetGraph_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('const float &', 'local_alpha'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_alpha", "ctx->nout.data.gpu_wr_ptr()", "ctx->value.data.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_all_cuda", [('const float &', 'local_alpha'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->nowned, local_alpha, ctx)"]),
], host = True),
Kernel("PageRank_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('int &', '__retval'), ('const float &', 'local_alpha'), ('float', 'local_tolerance'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("PageRank", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_alpha", "local_tolerance", "ctx->nout.data.gpu_wr_ptr()", "ctx->value.data.gpu_wr_ptr()"), "SUM"),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(retval.cpu_rd_ptr())"], parse = False),
], host = True),
Kernel("PageRank_all_cuda", [('int &', '__retval'), ('const float &', 'local_alpha'), ('float', 'local_tolerance'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["PageRank_cuda(0, ctx->nowned, __retval, local_alpha, local_tolerance, ctx)"]),
], host = True),
])
