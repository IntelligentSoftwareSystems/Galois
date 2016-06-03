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
CDeclGlobal([("unsigned int *", "P_NOUT", "")]),
CDeclGlobal([("float *", "P_RESIDUAL", "")]),
CDeclGlobal([("float *", "P_VALUE", "")]),
Kernel("ResetGraph", [G.param(), ('int ', 'nowned') , ('unsigned int *', 'p_nout'), ('float *', 'p_residual'), ('float *', 'p_value')],
[
ForAll("src", G.nodes(None, "nowned"),
[
CBlock(["p_value[src] = 0"]),
CBlock(["p_nout[src] = 0"]),
CBlock(["p_residual[src] = 0"]),
]),
]),
Kernel("InitializeGraph", [G.param(), ('int ', 'nowned') , ('const float ', 'local_alpha'), ('unsigned int *', 'p_nout'), ('float *', 'p_residual'), ('float *', 'p_value')],
[
ForAll("src", G.nodes(None, "nowned"),
[
CBlock(["p_value[src] = local_alpha"]),
CBlock(["p_nout[src] = graph.getOutDegree(src)"]),
If("p_nout[src] > 0",
[
CDecl([("float", "delta", "")]),
CBlock(["delta = p_value[src]*(1-local_alpha)/p_nout[src]"]),
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CBlock(["atomicAdd(&p_residual[dst], delta)"]),
]),
]),
]),
]),
Kernel("PageRank", [G.param(), ('int ', 'nowned') , ('const float ', 'local_alpha'), ('float', 'local_tolerance'), ('unsigned int *', 'p_nout'), ('float *', 'p_residual'), ('float *', 'p_value'), ('Any', 'any_retval')],
[
ForAll("src", G.nodes(None, "nowned"),
[
CDecl([("float", "residual_old", "")]),
CBlock(["residual_old = atomicExch(&p_residual[src], 0.0)"]),
CBlock(["p_value[src] += residual_old"]),
If("p_nout[src] > 0",
[
CDecl([("float", "delta", "")]),
CBlock(["delta = residual_old*(1-local_alpha)/p_nout[src]"]),
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CDecl([("float", "dst_residual_old", "")]),
CBlock(["dst_residual_old = atomicAdd(&p_residual[dst], delta)"]),
If("(dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance)",
[
CBlock(["any_retval.return_( 1)"]),
]),
]),
]),
]),
]),
Kernel("ResetGraph_cuda", [('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
Invoke("ResetGraph", ("ctx->gg", "ctx->nowned", "ctx->nout.gpu_wr_ptr()", "ctx->residual.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_cuda", [('const float &', 'local_alpha'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "local_alpha", "ctx->nout.gpu_wr_ptr()", "ctx->residual.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PageRank_cuda", [('int &', '__retval'), ('const float &', 'local_alpha'), ('float', 'local_tolerance'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
CBlock(["*(ctx->p_retval.cpu_wr_ptr()) = __retval"]),
CBlock(["ctx->any_retval.rv = ctx->p_retval.gpu_wr_ptr()"]),
Invoke("PageRank", ("ctx->gg", "ctx->nowned", "local_alpha", "local_tolerance", "ctx->nout.gpu_wr_ptr()", "ctx->residual.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()", "ctx->any_retval")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["__retval = *(ctx->p_retval.cpu_rd_ptr())"]),
], host = True),
])
