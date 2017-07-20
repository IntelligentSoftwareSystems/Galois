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
Kernel("ResetGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_nout'), ('float *', 'p_residual'), ('float *', 'p_value')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_value[src] = 0"]),
CBlock(["p_nout[src] = 0"]),
CBlock(["p_residual[src] = 0"]),
]),
]),
]),
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('const float ', 'local_alpha'), ('unsigned int *', 'p_nout'), ('float *', 'p_residual'), ('float *', 'p_value')],
[
CDecl([("float", "delta", "")]),
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_value[src] = local_alpha"]),
CBlock(["p_nout[src] = graph.getOutDegree(src)"]),
If("p_nout[src] > 0",
[
CBlock(["delta = p_value[src]*(1-local_alpha)/p_nout[src]"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CBlock(["atomicAdd(&p_residual[dst], delta)"]),
]),
),
]),
]),
Kernel("PageRank", [G.param(), ('unsigned int', '__nowned'), ('const float ', 'local_alpha'), ('float', 'local_tolerance'), ('unsigned int *', 'p_nout'), ('float *', 'p_residual'), ('float *', 'p_value')],
[
CDecl([("float", "residual_old", "")]),
CDecl([("float", "delta", "")]),
ForAll("wlvertex", WL.items(),
[
CDecl([("int", "src", "")]),
CDecl([("bool", "pop", "")]),
WL.pop("pop", "wlvertex", "src"),
If("pop", [
CBlock(["residual_old = atomicExch(&p_residual[src], 0.0)"]),
CBlock(["p_value[src] += residual_old"]),
If("p_nout[src] > 0",
[
CBlock(["delta = residual_old*(1-local_alpha)/p_nout[src]"]),
], [ CBlock(["pop = false"]), ]),
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CDecl([("float", "dst_residual_old", "")]),
CBlock(["dst_residual_old = atomicAdd(&p_residual[dst], delta)"]),
If("(dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance)",
[
WL.push("dst"),
]),
]),
),
]),
]),
Kernel("ResetGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("ResetGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->nout.gpu_wr_ptr()", "ctx->residual.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()")),
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
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "local_alpha", "ctx->nout.gpu_wr_ptr()", "ctx->residual.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_all_cuda", [('const float &', 'local_alpha'), ('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->nowned, local_alpha, ctx)"]),
], host = True),
Kernel("PageRank_cuda", [('const float &', 'local_alpha'), ('float', 'local_tolerance'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CBlock(["ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items)"]),
CBlock(["ctx->out_wl.will_write()"]),
CBlock(["ctx->out_wl.reset()"]),
Invoke("PageRank", ("ctx->gg", "ctx->nowned", "local_alpha", "local_tolerance", "ctx->nout.gpu_wr_ptr()", "ctx->residual.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()", "ctx->in_wl", "ctx->out_wl")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["ctx->out_wl.update_cpu()"]),
CBlock(["ctx->shared_wl->num_out_items = ctx->out_wl.nitems()"]),
], host = True),
])
