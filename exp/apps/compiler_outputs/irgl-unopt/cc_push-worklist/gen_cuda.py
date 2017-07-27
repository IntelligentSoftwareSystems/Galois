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
Kernel("InitializeGraph", [G.param(), ('unsigned int', '__nowned'), ('unsigned int', '__begin'), ('unsigned int', '__end'), ('unsigned int *', 'p_comp_current')],
[
ForAll("src", G.nodes("__begin", "__end"),
[
CDecl([("bool", "pop", " = src < __end")]),
If("pop", [
CBlock(["p_comp_current[src] = graph.node_data[src]"]),
]),
]),
]),
Kernel("ConnectedComp", [G.param(), ('unsigned int', '__nowned'), ('unsigned int *', 'p_comp_current')],
[
ForAll("wlvertex", WL.items(),
[
CDecl([("int", "src", "")]),
CDecl([("bool", "pop", "")]),
WL.pop("pop", "wlvertex", "src"),
If("pop", [
]),
UniformConditional(If("!pop", [CBlock("continue")]), uniform_only = False, _only_if_np = True),
ClosureHint(
ForAll("jj", G.edges("src"),
[
CDecl([("index_type", "dst", "")]),
CBlock(["dst = graph.getAbsDestination(jj)"]),
CDecl([("unsigned int", "new_dist", "")]),
CBlock(["new_dist = p_comp_current[src]"]),
CDecl([("unsigned int", "old_dist", "")]),
CBlock(["old_dist = atomicMin(&p_comp_current[dst], new_dist)"]),
If("old_dist > new_dist",
[
WL.push("dst"),
]),
]),
),
]),
]),
Kernel("InitializeGraph_cuda", [('unsigned int ', '__begin'), ('unsigned int ', '__end'), ('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
Invoke("InitializeGraph", ("ctx->gg", "ctx->nowned", "__begin", "__end", "ctx->comp_current.gpu_wr_ptr()")),
CBlock(["check_cuda_kernel"], parse = False),
], host = True),
Kernel("InitializeGraph_all_cuda", [('struct CUDA_Context *', 'ctx')],
[
CBlock(["InitializeGraph_cuda(0, ctx->nowned, ctx)"]),
], host = True),
Kernel("ConnectedComp_cuda", [('struct CUDA_Context *', 'ctx')],
[
CDecl([("dim3", "blocks", "")]),
CDecl([("dim3", "threads", "")]),
CBlock(["kernel_sizing(blocks, threads)"]),
CBlock(["ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items)"]),
CBlock(["ctx->out_wl.will_write()"]),
CBlock(["ctx->out_wl.reset()"]),
Invoke("ConnectedComp", ("ctx->gg", "ctx->nowned", "ctx->comp_current.gpu_wr_ptr()", "ctx->in_wl", "ctx->out_wl")),
CBlock(["check_cuda_kernel"], parse = False),
CBlock(["ctx->out_wl.update_cpu()"]),
CBlock(["ctx->shared_wl->num_out_items = ctx->out_wl.nitems()"]),
], host = True),
])
