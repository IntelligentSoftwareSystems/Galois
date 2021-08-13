Author: Loc Hoang, <l_hoang@utexas.edu>

Best viewed with a Markdown viewer due to Latex + formatting.

This file's sections are ordered such that you can read from
top to bottom and still get a decent understanding of the
pieces of `libgnn`. As such, independent portions are near the
top.

This file is being written so that whoever works on this code in the
future has a general idea what contributions I've made to the code
and how the gnn branch differs from master. Some of these changes
need to get merged into master in the future. It also allows me
to take stock of the changes/implementation choices I've made
in the past year.

# CuSP Changes

Variants of the regular partitions were added to allow training
nodes to be partitioned relatively evenly among machines rather
than having CVC/OEC use a regular block partition over all nodes (which
would ignore the train/val/test split).

This causes some weird effects when this version's CuSP is used outside
of GNNs or if the training boundaries are not hardcoded (e.g., if
the training boundaries are unknown, a segfault can occur). Some care
will be needed to make this integration more clean.

# Gluon Changes

Many changes occurred to Gluon to optimize for the vector communication
case. A few of them are listed below.

* Serialize/deserialize **directly** to/from the serialization and
deserialization buffers. This eliminates a large amount of redundant
copying from original source to vector to buffer (and in the reverse)
which is incredibly important for performance when communicating vectors.
Something important to also take away from this experience is that
if you have a vector of vectors, serializing each vector individually
into the buffer is a very bad idea: care should be taken to make
it so that you can serialize as much data as possible in one go.

* QoL change: way to disable Gluon timers with a variable change/flag.

* Method to swap out mirror handshake since this is used by subgraph
code to avoid sending messages to inactive mirrors.

* Hochan ported large message handling from KatanaGraph into Galois.
This involved changing the serialization buffers among other things.

# GNN Optimizers

Only one that exists is the ADAM optimizer. Note that each
layer has its own moments and does not share them (this may or
may not be standard; I'm not sure).

All hosts will see the same gradients due to synchronization,
so all hosts should end up making the same changes to the weights.

# Layers

Each layer inherits from a `GNNLayer`  class which has common functionality
like weight allocation, output allocation, etc. The children classes
can add more things to it; for example, SAGE adds weights for the
concatenated feature and intermediates for intermediate calculation
(also reused in backward prop).

One thing to note is that the backward output matrix (used to output
gradients during the backward phase) is **not** a completely independent
piece of memory: it is the **memory used by the forward output of
the layer that came before it**. The reason for this is that doing it
this way saves a very large amount of memory, especially in a full batch
setting where the number of nodes (multiplied by features/hidden feature
size) can grow very large. **Be very careful about this as it means that
you cannot reuse the output matrix from the forward pass after it
has been overwritten.** This results in some rather convoluted logic that
you may find in the code. It also means that **whenever an output matrix
is resized for any reason, the pointers that each layer holds MUST
be updated, or you will get undefined behavior**.

## Softmax Layer

Runs a softmax on each individual row, gets the highest value,
compares with ground truth, gets loss.

Note that the **forward and backward output matrix are shared** in this
layer, so be careful with the assumptions made after the backward
step is run (because the forward output will no longer be accessible
after the backward step; this is why the accuracy check in the
code has to occur before backward is called).

Regarding the backward step: it turns out that for single class
classification, the gradient if the answer is wrong is simply
the softmax value itself, and if the answer is right, then its 
the softmax value - 1. This has the advantage of being very
numerically stable as well.

Things are slightly more complicated for the multi-class case; some
investigation needs to be done to figure this out.

## SAGE Layer (and GCN Layer by Extension)

### ReLU Activation and Overwriting of the Forward Matrix

ReLU activation is used by the compute layers: if the value
is greater than 0, it is kept, else it is discarded.

Because the forward output matrix gets overwritten during
the backward step and because the derivative of the 
ReLU operation requires knowledge of what elements were
affected by the ReLU, the system must *track* which
elements were not set to 0 using a bitmask. This
mask is used during the backward phase to keep gradients
only if their values corresponding to that gradient
were originally greater than 0, and it works even
if the original forward matrix has been overwritten.

### Row Dimensions and Active Portions of Matrices

An optimal version of a normal GNN should make it so that
the number of active rows decreases as execution progresses
through the layers of the GNN: the last layer's active
rows in the feature matrix should be *only* the seed
nodes (i.e., nodes that are being predicted): keeping
all nodes up to date is a waste of compute.

The number of active nodes at the beginning of a GNNs
should be all nodes involved in the k-hop neighborhood
of the seed nodes. The next layer should remove
the kth hop from the active nodes; the layer after,
the (k-1)th hop, and so on. This can be accomplished
relatively easily without disrupting the contiguous
feature matrix by making sure that the nodes that will
be dropped are in the suffix of matrix in the order
that they will be dropped from the bottom. Then,
to drop them, the code just changes the number of input
rows for the layer so that any loops/matrix multiplies
will only look at the relevant row prefix.

In a distributed setting, the active nodes of a particular
layer should be *shared* across all hosts; a host should not
drop a node if it is being used somewhere else *and* if
the node in question has a contribution to it (i.e.,
has edges or is the master proxy).

### SAGE's Concatenation of Input Features

The GraphSAGE model concatenates the input feature to the aggregated
feature vector on each node after aggregation which doubles
the length of the vector. Actually doing this in the feature
matrix is not great as it would mean that the original weight
matrix needs to double in size, and additional space would have
to be allocated on top of the existing input features
with the aggregated copied over to it. 

Instead of doing this, you can allocate a separate weight matrix
of the same size as the original, multiply the original input
features with that new weight matrix, and sum it up to the final
output matrix. The result is exactly the same as if the input
feature was concatenated to the aggregated features then
multiplied with a weight matrix with double the number of rows.
(work it out mathematically; it's the same)

### Intermediates and Flipping Aggregation/Linear XForm: Basics

The GNN computation in SAGE is two-step: aggregation
followed by linear transform (more steps if dropout is enabled):
an intermediate matrix is required to store the result of the first
step for use in the next step. Additionally, keeping this
intermediate result around in memory significantly speeds up
the backward step which can use it to derive gradients.
Therefore, the SAGE layer must allocate space for the intermediate.

The size of the intermediate changes depending on if you do
linear xform before aggregation; this is done if doing
the linear xform reduces the column dimension as it makes
the aggregation aggregate on smaller feature vector sizes (which
speeds up computation overall in general). It helps to understand
how the dimensions change after aggregation and after linear
xform. Say the input matrix is IR by IC (input row by input column).

* Aggregation only needs to occur for the nodes that will
be active in the next layer, i.e. the *output rows* (OR). Therefore,
after aggregation, the rows of the matrix go from IR to OR.

* Linear transform changes the number of columns to output columns (OC).
Therefore, after linear xform, IC turns to OC.

After both operations, the output matrix to the next layer is the
expected OR by OC. Depending on which one occurs first, 
the code generates an intermediate of OR by IC *or* IC by OC.
(more than one may be needed if dropout is used as that generates
a new dropout matrix).

### Intermediates and Flipping Aggregation/Linear XForm: Backward Pass

The computation of a SAGE layer is the following in matrix
terms where $T$ is the graph topology, $F$ is features,
and the $W$s are the two weight matrices (one for aggregated
value, other for concatenated vector).

$TFW_1 + FW_2 = O$

The gradients we want are $W_{1,2}'$ and $F'$ to pass back to the next layer in
the backward phase. We have the gradient $O'$. The method in which this occurs
depends on the order of aggregation/xform in the forward phase.

First, $FW_2$. One can derive one part of $F'$ (the other part
is from the first term) and $W_{2}'$. $F' = O'(W_2)^T$ and $W_{2}' = F^T O'$.

Next, $TFW_1$.

* If aggregation occurs first, we have $(TF)$ in an intermediate
matrix.  The $W_{1}'$ gradient is $W_{1}' = (TF)^{T}O'$. To get one part of
$F'$, we do $O' W_{1}^{T} = (TF)'$ followed by $T^T (TF)' = F'$.
* If xform occurs first, $(FW_1)$ is in the intermediate matrix.
To get $F'$, $T^T O' = (FW_{1})'$, followed by $(FW_{1})' (W_{1})^T = F'$.
The weight gradient is $W_{1}' = F^T (FW_{1})'$.

The $F'$ gradient from the two terms ($TFW_1$ and $FW_2$) can be summed
together.

### Masking Out Non-Masters in Distributed Setting

In a distributed setting, all hosts need to see the same gradient
computed in the backward phase so that the weights can all be updated
in the same manner to keep consistency across hosts. This can
be accomplished by synchronizing appropriately and making
sure that a gradient computation isn't accounted for more than
once globally.

For $F'$, keeping it consistent simply means making sure that all
hosts compute all the required rows. This is doable if a host knows
what proxies it owns are active in the global subgraph being operated
on and makes sure that it has the most up-to-date value for that proxy's
gradient at all times. For example, since all hosts have a copy of the
weights, in order to get the gradients for $F'$, all a host needs
is to make sure $O'$ contains the gradients for local proxies
active in a particular layer (even if they aren't part of that
host's seed nodes). In this way, all hosts *recompute* the same gradient
required for a proxy.

For $W'$, each node contributes a gradient to it. A node is
replicated across hosts via proxies; unlike the previous case,
however, a *sync* of weight gradients occurs across all hosts because
not all hosts have all proxies, and in this case, you need the
contribution of all nodes and not just the ones you have proxies
of, so you do **not** want a node's gradient to be computed more than
once across all hosts. Therefore, when doing computation involving
the weight gradient, a node's contribution should only be computed
once **by the owner/master of that node**. Therefore, non-masters
on hosts **need to be masked when computing $W'$**.
This presents a problem implementation wise: masking non-masters
is an in-place operation since you do not want to allocate
new memory, so some care needs to be taken on which matrices to mask
as well as when to mask them since $F'$ computation requires *non-masked*
matrices. This is the reason for the very convoluted logic in the
backward pass in the code that will need to be cleaned up or
redesigned at some point.
It might be possible to play a similar trick to active row prefixing
where non-masters are placed lower in the rows so that "masking"
can occur by changing the row count, but I believe I tried
this and ran into issues with non-contiguity of masters/mirrors.

Below is the masking logic used by the current code:

```
Calculate W2' using masked input or masked gradients (mask required else overcount,
if not layer 0 then can mask input, else mask gradient)

if (xform before agg)
  Calculate (FW1)' by tranpose aggregating gradients
  Mask out the non-masters in feature matrix F if not layer 0, else mask FW1
  Calculate W1' using F^T and (FW1)' (one of which is masked)
  Calculate F' from W1 by using (FW1)', W1^T and W2^T (masked FW1 won't occur here,
  because this is only required if layer isn't 0)
else
  Mask F if not layer 0, else mask gradient
  Get F' from W2 by multiplying O' with W2 (no masks allowed here)
  Mask TF^T if not layer 0 (because O' won't be masked in that case)
  Get W1' by multiplying TF^T with O' (one will be masked)
  Get F' from W1 by (1) multipling O' with W1^T then (2) transpose aggregate to get F'
  (none of the ops above should be masked)
```

The above isn't the neatest explanation of things, but essentially,
anything involving a W' calculation requires one of the operands
to have masked non-masters. Layer 0 is special because you
can't mask the inputs there as those are the inputs used at
the beginning of an epoch.

### Regarding Dropout

The way that dropout works is that random parts of the input
are set to 0 for that particular batch.
The ones set to 0 need to be memorized so that the backward
pass can correctly compute the derivative.

Dropout currently **does not work in a distributed setting**: the problem
is that each host may dropout different weights due to the nature
of RNG, leading to divergence on each host. One way to avoid
this is to make it so each host dropouts a particular portion only and
synchronize this choice. This has not been implemented efficiently (yet?).
**I have not kept this code up-to-date as well** as all runs I've been
doing are without dropout.

*Therefore, it's probably better not to use it for the time being.*

# Graph Neural Network

`GraphNeuralNetwork.{cpp/h}` is the main class which runs the
graph neural network. It creates the layers and chains their outputs
together to create the network flow.

## Constructor

1) Creates the intermediate layers. See the section on Layers to get an
idea of what is done.
Typically, activation is activated for compute layers except for the last
layer: activation is typically disabled for that layer for accuracy
reasons (running activation on the final output layer messes with
predictions).

2) If minibatching is enabled, create minibatch generators.

3) Create the output layer (Softmax is the only one that works right now,
but Sigmoid is required for multi-class classification).

## Training Flow

There are a few scenarios based on if training and testing minibatching
is enabled or not. These are not necessarily the most optimal things to
do (e.g., you never want the entire graph to participate in training;
only k-hop neighborhood is required).

1) No training/testing minibatch -> the entire graph participates in training.

2) Training minibatch but no test minibatch -> k-hop neighborhood only, but
space required for entire graph is allocated (inefficient, should only need
k-hop neighborhood of test nodes)

3) Train/test minibatching -> k-hop neighborhood subgraphs only, and space
for them is allocated on demand rather than worst case entire graph.

Note that because of the way the code works, if you want to do an *efficient*
full-batch no sampling run, you should specify very large numbers for the train
and test minibatches so that the efficient code path is taken. Due to the
way the design is at the moment it will **inefficiently regenerate
the k-hop full batch train/test subgraphs when they are used**: this
need to be fixed in a future redesign where multiple subgraphs can be
swapped among.

If a k-hop subgraph needs to be generated, it's generated with the following
flow:

1) Choose the seed nodes (i.e., nodes that will have their output compared
to ground truth to potentially get loss/gradients to backpropagate)

2) From seed nodes, sample a few edges OR if not sampling, choose all
of them. Activate the destination nodes, communicate this, repeat
for k hops.

3) Correct layer dimensions based on subgraph/number of nodes at
each layer (reduce memory AND compute footprint).

4) Generate subgraph (see subgraph construction section).

5) Do inference and back prop, update weights, repeat.
The way this works is relatively simple: the code loops
through each layer and calls the forward or backward pass function
on it.

Depending on how the test interval is set, between each epoch 
a test subgraph may be used to check test accuracy.
The flaw with the current design is that the graph object is
only aware of one 'graph' at any one point, meaning the code
has to be very careful to generate the right graph (train/test)
for use at the right time.

Note that the `kBatch` mode used in the Train code refers to
a status that is set on nodes based on the minibatch and only
includes *local seed nodes*, so keep this in mind when using it (there
have been unintentional problems where I assumed `kBatch` meant
more than just local seed nodes). The main reason for this is
that it helps to distinguish local and global seed nodes to avoid 
over-calculating gradients.

# GNN Graph

`GNNGraph.{cpp/h}` is responsible for reading in the graph topology,
labels, and features. Topology is read/partitioned via the CuSP
infrastructure. Each host reads labels for nodes it owns; same with
features (right now it's pretty inefficient as all hosts read the entire
file; some better way should probably be come up with).

It is responsible for the synchronization substrate: Gluon is initialized
on the partitioned graph. Normally sync occurs on the node data of the graph,
but the node data in GNN case is a feature vector. To get around sync
structure limitations, a global pointer is set to point to the feature
matrix array (along with some other globals) so that the sync structure
can know how to access it.

There are sync structures for global degrees and aggregation mainly.
If a subgraph is used, things change slightly (see subgraph section)

The class provides functions to get degrees and also holds the minibatch
generator. It also holds one `GNNSubgraph` object if a subgraph is being used
(this is a limitation; there can only be one active subgraph at any one point).
If the subgraph is active and the flag for the subgraph is on, then all
user-facing functions on the `GNNGraph` object will access the *subgraph*
instead of the original graph. **Be very careful with this and make sure the
graph is in the right mode that you intend it to be.**

# Subgraph Construction

Subgraphs are created by the sampling/minibatch infrastructure:
a few nodes are marked "active" along with edges, and
the program compiles these chosen nodes/edges into a separate
CSR for use during execution. There are a few implementation details
during this process that will be documented here.

## Code Structure

The current implementation in Galois has a Subgraph class
contained by the GNNGraph class. The subgraph is enabled
by a flag which alters GNNGraph calls to direct to the
subgraph instead.

Optimally, we want to be able to work with many subgraphs
at once; this design makes it difficult to do so as
only 1 subgraph is contained by on GNNGraph. It would
probably be possible to extend this design and have GNNGraph
expose a subgraph switcher or something of the sort so that
it isn't tied directly to the class.

## Sampling

The "activeness" of a node is marked on the node itself as a flag.
In addition to this, the layer number in which a node is added
is noted as well (the reason for this will be apparent later).

Each edge has two variables associated with it: a normal flag
saying if it has been sampled in any layer, and a bitset saying
which layers the edge has been sampled in. This is because
an edge once sampled is not necessarily sampled in *all* layers:
it may be sampled in only a single layer (or many layers),
and this info needs to be known when iterating over the edges
to keep things correct.

In addition, the degree of a node for each sampled phase locally
is kept track of. At the end of all sampling, the degrees
of the nodes at each layer are synchronized among all hosts.
This is required because normalization in aggregation uses 
the subgraph degrees (this is actually quite annoying runtime
wise as it adds this extra degree sync step).

## Construction Steps

The steps in subgraph construction are the following:

1) Create the local ID to subgraph ID mapping (and vice versa)
2) Count degrees for the sampled vertices in order to construct
the CSR; this includes edges that may not always be active.
3) Create the CSR using the degrees.
4) Create the local subgraph features matrix by copying
them over from the original feature matrix.

In order to make row elimination easier, 
the SID of the vertices are ordered such that seed nodes are
first, the 1-hop samples next, then 2-hops, 3-hops, etc.
This makes it easy to eliminate vertices that aren't used after
a certain point by changing the row dimensions used by multiplies/
aggregations. Master nodes that are also seed nodes always occupy
the first SIDs so that it's easy to loop through master nodes only.
Other master nodes may end up with non-contiguous SIDs as they
may become active in different layers; to track these masters
for masking later, a bitset is maintained.
Counts as to how many nodes are in each layer have to be
compiled so this process can be done in parallel. An on_each
loop is used to get SIDs in parallel.

In addition, nodes that (1) are not master proxies and (2) do
not have any outgoing or incoming edges are eliminated from
the local subgraph. This is because some proxies do not have
edges on some hosts even if they do on other hosts, so even
if they become active, they do not change the outcome of computation
and actually add unnecessary overhead. **This dead mirror
removal is extremely important for performance.** Implementation
wise it is done by keeping a "definitely active" flag which
will only mark proxies that definitely have an edge connecting
them or proxies that are masters.

Degree counting and graph construction proceed as normal: count
degrees, do a prefix sum, create the CSR. One thing to note is
that the CSC is also created in order to do the backward aggregation
step. The data which says which layers an edge is active in is
pointed to by the newly constructed graph.

## Synchronization when Subgraphs Exist

### Mirror Regeneration

Some mirrors on a local host may be inactive in the subgraph because
they were not sampled. The subgraph code can create a new mirror
node mapping that Gluon can swap out for each subgraph.

This has its own overhead, and from some experiments in the
past this doesn't significantly affect performance, but it's
done anyways.

### GID to SID

Gluon memoizes GID-LID handshakes on each host to avoid the need
to send IDs along with messages. This means that if a subgraph is being
synchronized, another conversion to SIDs must occur. There need
to be sampled graph versions of the sync structures that use
a mapping from LID to SID in order to save the updates to the correct
memory locations.

Sometimes, due to the way Gluon works, a node that isn't part of the
active subgraph may have its data queried for extract/update. The sync
structure must account for this and check if such data is being accessed
so that it can avoid seg-faulting.

# Minibatch Generator

`MinibatchGenerator.{cpp/h}` takes the list of training/test nodes on
a single host and gives the user an interface for getting the nodes
in batches at a time. This is used to do minibatching of nodes across
hosts; each host picks the same number at a time before the beginning
of minibatch.

# Other (Dead) Files/Code

`DistributedMinibatchTracker` was created to track variable number
of seed nodes on each host to make the sampling more like single-host
sampling. This was deprecated for a new functionality in the `MinibatchGenerator`
which does it in a much more sane manner by having all hosts see the same
global sequence of nodes to choose and moving the window locally on each
host (this can result in imbalanced seeds).

A lot of the existing layers have not been kept up-to-date due to the rapid
development process on minibatching/sampling. Only the SAGE layer and Softmax
Layer are guaranteed to be functional as those are the ones most
of the runs have been on.

There is an experimental implementation of something known as "sampled views"
in which an explicit subgraph isn't constructed; a mask is used instead.
Performance wise this did not do too well, so the code has been abandoned
and is not guaranteed to work.

# Regarding GPU Code

It has been a while since I worked on the GPU code, but the idea is essentially
to pre-allocate the same data that you would have allocated on the CPU
and use those pointers instead of CPU pointers.

Some updates will need to be made in order to do dynamic resizing of the
data depending on the size of the minibatch. The best way to avoid this
in general, though, is to just allocate space for the test subgraph's
k-hops since that is likely to be more expensive than whatever
the minibatch size for the train nodes are (unless it's all nodes).