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
Kernel("InitializeGraph", [G.param(), ('int ', 'nowned') , ('const float ', 'local_alpha'), ('int *', 'p_nout'), ('float *', 'p_value')],
[
ForAll("src", G.nodes(None, "nowned"),
[
CBlock(["p_value[src] = 1.0 - local_alpha"]),
ForAll("nbr", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(nbr)"]),
CBlock(["atomicAdd(&p_nout[dst], 1)"]),
]),
]),
]),
Kernel("PageRank_pull", [G.param(), ('int ', 'nowned') , ('const float ', 'local_alpha'), ('float', 'local_tolerance'), ('int *', 'p_nout'), ('float *', 'p_value')],
[
ForAll("wlvertex", WL.items(),
[
CDecl([("int", "src", "")]),
CDecl([("bool", "pop", "")]),
WL.pop("pop", "wlvertex", "src"),
CDecl([("float", "sum", "")]),
CBlock(["sum = 0"]),
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
CDecl([("float", "pr_value", "")]),
CBlock(["pr_value = sum*(1.0 - local_alpha) + local_alpha"]),
CDecl([("float", "diff", "")]),
CBlock(["diff = fabs(pr_value - p_value[src])"]),
If("diff > local_tolerance",
[
CBlock(["p_value[src] = pr_value"]),
WL.push("src"),
]),
]),
]),
Kernel("InitializeGraph_cuda", [('const float &', 'local_alpha'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "local_alpha", "ctx->nout.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("PageRank_pull_cuda", [('const float &', 'local_alpha'), ('float', 'local_tolerance'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(ctx->gg, blocks, threads)"]),
CBlock(["ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items)"]),
CBlock(["ctx->out_wl.will_write()"]),
CBlock(["ctx->out_wl.reset()"]),
Invoke("PageRank_pull", ("ctx->gg", "ctx->nowned", "local_alpha", "local_tolerance", "ctx->nout.gpu_wr_ptr()", "ctx->value.gpu_wr_ptr()", "ctx->in_wl", "ctx->out_wl")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["ctx->out_wl.update_cpu()"]),
CBlock(["ctx->shared_wl->num_out_items = ctx->out_wl.nitems()"]),
], host = True),
])
