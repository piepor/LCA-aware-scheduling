??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878??
z
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@H* 
shared_namedense_54/kernel
s
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes

:@H*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:H*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_14/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_14/lstm_cell_14/kernel
?
/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_14/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*6
shared_name'%lstm_14/lstm_cell_14/recurrent_kernel
?
9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_14/lstm_cell_14/recurrent_kernel*
_output_shapes
:	@?*
dtype0
?
lstm_14/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_14/lstm_cell_14/bias
?
-lstm_14/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@H*'
shared_nameAdam/dense_54/kernel/m
?
*Adam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/m*
_output_shapes

:@H*
dtype0
?
Adam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_54/bias/m
y
(Adam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/m*
_output_shapes
:H*
dtype0
?
"Adam/lstm_14/lstm_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_14/lstm_cell_14/kernel/m
?
6Adam/lstm_14/lstm_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_14/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_14/lstm_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m
?
@Adam/lstm_14/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_14/lstm_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_14/lstm_cell_14/bias/m
?
4Adam/lstm_14/lstm_cell_14/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_14/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@H*'
shared_nameAdam/dense_54/kernel/v
?
*Adam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/v*
_output_shapes

:@H*
dtype0
?
Adam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_54/bias/v
y
(Adam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/v*
_output_shapes
:H*
dtype0
?
"Adam/lstm_14/lstm_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_14/lstm_cell_14/kernel/v
?
6Adam/lstm_14/lstm_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_14/lstm_cell_14/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_14/lstm_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v
?
@Adam/lstm_14/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_14/lstm_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_14/lstm_cell_14/bias/v
?
4Adam/lstm_14/lstm_cell_14/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_14/lstm_cell_14/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?"
value?"B?" B?"
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
	cell


state_spec
_inbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
|
_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratemCmDmEmFmGvHvIvJvKvL
#
0
1
2
3
4
#
0
1
2
3
4
 
?
metrics
	variables
 layer_metrics
!layer_regularization_losses

"layers
trainable_variables
regularization_losses
#non_trainable_variables
 
~

kernel
recurrent_kernel
bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
 
 

0
1
2

0
1
2
 
?
(metrics
	variables
)layer_metrics
*layer_regularization_losses

+layers

,states
trainable_variables
regularization_losses
-non_trainable_variables
 
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
.metrics
	variables
trainable_variables
/layer_regularization_losses

0layers
1layer_metrics
regularization_losses
2non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_14/lstm_cell_14/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_14/lstm_cell_14/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_14/lstm_cell_14/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

30
41
 
 

0
1
 

0
1
2

0
1
2
 
?
5metrics
$	variables
%trainable_variables
6layer_regularization_losses

7layers
8layer_metrics
&regularization_losses
9non_trainable_variables
 
 
 

	0
 
 
 
 
 
 
 
4
	:total
	;count
<	variables
=	keras_api
D
	>total
	?count
@
_fn_kwargs
A	variables
B	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

<	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

A	variables
~|
VARIABLE_VALUEAdam/dense_54/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_14/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_14/lstm_cell_14/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_14/lstm_cell_14/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_54/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_14/lstm_cell_14/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_14/lstm_cell_14/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_14/lstm_cell_14/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_14_inputPlaceholder*+
_output_shapes
:?????????0*
dtype0* 
shape:?????????0
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_14_inputlstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biasdense_54/kerneldense_54/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_499873
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOp9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOp-lstm_14/lstm_cell_14/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_54/kernel/m/Read/ReadVariableOp(Adam/dense_54/bias/m/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_14/kernel/m/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_14/bias/m/Read/ReadVariableOp*Adam/dense_54/kernel/v/Read/ReadVariableOp(Adam/dense_54/bias/v/Read/ReadVariableOp6Adam/lstm_14/lstm_cell_14/kernel/v/Read/ReadVariableOp@Adam/lstm_14/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_14/lstm_cell_14/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_501091
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_54/kerneldense_54/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biastotalcounttotal_1count_1Adam/dense_54/kernel/mAdam/dense_54/bias/m"Adam/lstm_14/lstm_cell_14/kernel/m,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m Adam/lstm_14/lstm_cell_14/bias/mAdam/dense_54/kernel/vAdam/dense_54/bias/v"Adam/lstm_14/lstm_cell_14/kernel/v,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v Adam/lstm_14/lstm_cell_14/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_501173??
?
?
.__inference_sequential_34_layer_call_fn_499848
lstm_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_4998352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_14_input
?
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_499835

inputs
lstm_14_499822
lstm_14_499824
lstm_14_499826
dense_54_499829
dense_54_499831
identity?? dense_54/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_499822lstm_14_499824lstm_14_499826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4997122!
lstm_14/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_54_499829dense_54_499831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_4997522"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_54/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
while_cond_499324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_499324___redundant_placeholder04
0while_while_cond_499324___redundant_placeholder14
0while_while_cond_499324___redundant_placeholder24
0while_while_cond_499324___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?9
?

__inference__traced_save_501091
file_prefix.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableopD
@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop8
4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_54_kernel_m_read_readvariableop3
/savev2_adam_dense_54_bias_m_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_14_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_14_bias_m_read_readvariableop5
1savev2_adam_dense_54_kernel_v_read_readvariableop3
/savev2_adam_dense_54_bias_v_read_readvariableopA
=savev2_adam_lstm_14_lstm_cell_14_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_14_lstm_cell_14_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2475dc303f51429b81ed7a80a7982eef/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableop@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_54_kernel_m_read_readvariableop/savev2_adam_dense_54_bias_m_read_readvariableop=savev2_adam_lstm_14_lstm_cell_14_kernel_m_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_14_lstm_cell_14_bias_m_read_readvariableop1savev2_adam_dense_54_kernel_v_read_readvariableop/savev2_adam_dense_54_bias_v_read_readvariableop=savev2_adam_lstm_14_lstm_cell_14_kernel_v_read_readvariableopGsavev2_adam_lstm_14_lstm_cell_14_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_14_lstm_cell_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@H:H: : : : : :	?:	@?:?: : : : :@H:H:	?:	@?:?:@H:H:	?:	@?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@H: 

_output_shapes
:H:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:%	!

_output_shapes
:	@?:!


_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@H: 

_output_shapes
:H:%!

_output_shapes
:	?:%!

_output_shapes
:	@?:!

_output_shapes	
:?:$ 

_output_shapes

:@H: 

_output_shapes
:H:%!

_output_shapes
:	?:%!

_output_shapes
:	@?:!

_output_shapes	
:?:

_output_shapes
: 
?

?
lstm_14_while_cond_500099,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1D
@lstm_14_while_lstm_14_while_cond_500099___redundant_placeholder0D
@lstm_14_while_lstm_14_while_cond_500099___redundant_placeholder1D
@lstm_14_while_lstm_14_while_cond_500099___redundant_placeholder2D
@lstm_14_while_lstm_14_while_cond_500099___redundant_placeholder3
lstm_14_while_identity
?
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2
lstm_14/while/Lessu
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_14/while/Identity"9
lstm_14_while_identitylstm_14/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_sequential_34_layer_call_fn_499817
lstm_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_4998042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_14_input
?l
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_500032

inputs7
3lstm_14_lstm_cell_14_matmul_readvariableop_resource9
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource8
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource
identity??lstm_14/whileT
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_14/Shape?
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stack?
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1?
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2?
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slicel
lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/mul/y?
lstm_14/zeros/mulMullstm_14/strided_slice:output:0lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/mulo
lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros/Less/y?
lstm_14/zeros/LessLesslstm_14/zeros/mul:z:0lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/Lessr
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/packed/1?
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros/packedo
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros/Const?
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zerosp
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/mul/y?
lstm_14/zeros_1/mulMullstm_14/strided_slice:output:0lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/muls
lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros_1/Less/y?
lstm_14/zeros_1/LessLesslstm_14/zeros_1/mul:z:0lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/Lessv
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/packed/1?
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros_1/packeds
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros_1/Const?
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zeros_1?
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/perm?
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:0?????????2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1?
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stack?
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1?
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2?
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1?
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_14/TensorArrayV2/element_shape?
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensor?
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stack?
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1?
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2?
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_14/strided_slice_2?
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp?
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul?
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul_1?
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/add?
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/BiasAddz
lstm_14/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/lstm_cell_14/Const?
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dim?
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_14/lstm_cell_14/split?
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Sigmoid?
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_1?
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul?
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu?
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_1?
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/add_1?
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_2?
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu_1?
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_2?
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_14/TensorArrayV2_1/element_shape?
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2_1^
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/time?
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counter?
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_14_while_body_499941*%
condR
lstm_14_while_cond_499940*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_14/while?
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStack?
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_14/strided_slice_3/stack?
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1?
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2?
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_14/strided_slice_3?
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/perm?
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtime?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

:@H*
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMul lstm_14/strided_slice_3:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_54/BiasAdd}
IdentityIdentitydense_54/BiasAdd:output:0^lstm_14/while*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2
lstm_14/whilelstm_14/while:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?D
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_499262

inputs
lstm_cell_14_499180
lstm_cell_14_499182
lstm_cell_14_499184
identity??$lstm_cell_14/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_499180lstm_cell_14_499182lstm_cell_14_499184*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_4988662&
$lstm_cell_14/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_499180lstm_cell_14_499182lstm_cell_14_499184*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_499193*
condR
while_cond_499192*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_14/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?%
?
while_body_499193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_14_499217_0
while_lstm_cell_14_499219_0
while_lstm_cell_14_499221_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_14_499217
while_lstm_cell_14_499219
while_lstm_cell_14_499221??*while/lstm_cell_14/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_499217_0while_lstm_cell_14_499219_0while_lstm_cell_14_499221_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_4988662,
*while/lstm_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_14_499217while_lstm_cell_14_499217_0"8
while_lstm_cell_14_499219while_lstm_cell_14_499219_0"8
while_lstm_cell_14_499221while_lstm_cell_14_499221_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?9
?
while_body_500442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?l
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_500191

inputs7
3lstm_14_lstm_cell_14_matmul_readvariableop_resource9
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource8
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource
identity??lstm_14/whileT
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_14/Shape?
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stack?
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1?
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2?
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slicel
lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/mul/y?
lstm_14/zeros/mulMullstm_14/strided_slice:output:0lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/mulo
lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros/Less/y?
lstm_14/zeros/LessLesslstm_14/zeros/mul:z:0lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/Lessr
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros/packed/1?
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros/packedo
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros/Const?
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zerosp
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/mul/y?
lstm_14/zeros_1/mulMullstm_14/strided_slice:output:0lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/muls
lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_14/zeros_1/Less/y?
lstm_14/zeros_1/LessLesslstm_14/zeros_1/mul:z:0lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/Lessv
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_14/zeros_1/packed/1?
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros_1/packeds
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros_1/Const?
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/zeros_1?
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/perm?
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:0?????????2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1?
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stack?
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1?
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2?
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1?
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_14/TensorArrayV2/element_shape?
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensor?
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stack?
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1?
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2?
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_14/strided_slice_2?
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp?
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul?
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/MatMul_1?
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/add?
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_14/lstm_cell_14/BiasAddz
lstm_14/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/lstm_cell_14/Const?
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dim?
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_14/lstm_cell_14/split?
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Sigmoid?
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_1?
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul?
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu?
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_1?
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/add_1?
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_14/lstm_cell_14/Sigmoid_2?
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/Relu_1?
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_14/lstm_cell_14/mul_2?
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_14/TensorArrayV2_1/element_shape?
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2_1^
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/time?
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counter?
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*%
bodyR
lstm_14_while_body_500100*%
condR
lstm_14_while_cond_500099*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_14/while?
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStack?
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_14/strided_slice_3/stack?
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1?
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2?
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_14/strided_slice_3?
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/perm?
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtime?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

:@H*
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMul lstm_14/strided_slice_3:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_54/BiasAdd}
IdentityIdentitydense_54/BiasAdd:output:0^lstm_14/while*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2
lstm_14/whilelstm_14/while:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
.__inference_sequential_34_layer_call_fn_500206

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_4998042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?W
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500855

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_500770*
condR
while_cond_500769*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::2
whilewhile:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
-__inference_lstm_cell_14_layer_call_fn_500996

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_4988992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????@:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?
?
(__inference_lstm_14_layer_call_fn_500549
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4993942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_499626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_499626___redundant_placeholder04
0while_while_cond_499626___redundant_placeholder14
0while_while_cond_499626___redundant_placeholder24
0while_while_cond_499626___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_500289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?D
?
lstm_14_while_body_500100,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0A
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0@
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor=
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource?
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource>
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource??
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItem?
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_14/while/lstm_cell_14/MatMul?
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_14/while/lstm_cell_14/MatMul_1?
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_14/while/lstm_cell_14/add?
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_14/while/lstm_cell_14/BiasAdd?
 lstm_14/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_14/while/lstm_cell_14/Const?
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dim?
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_14/while/lstm_cell_14/split?
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_14/while/lstm_cell_14/Sigmoid?
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_1?
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_14/while/lstm_cell_14/mul?
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_14/while/lstm_cell_14/Relu?
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_1?
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/add_1?
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_2?
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_14/while/lstm_cell_14/Relu_1?
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_2?
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_14/while/TensorArrayV2Write/TensorListSetIteml
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add/y?
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/addp
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add_1/y?
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1v
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_14/while/Identity?
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1x
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2?
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3?
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_4?
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_5"9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"?
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?

?
lstm_14_while_cond_499940,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1D
@lstm_14_while_lstm_14_while_cond_499940___redundant_placeholder0D
@lstm_14_while_lstm_14_while_cond_499940___redundant_placeholder1D
@lstm_14_while_lstm_14_while_cond_499940___redundant_placeholder2D
@lstm_14_while_lstm_14_while_cond_499940___redundant_placeholder3
lstm_14_while_identity
?
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2
lstm_14/while/Lessu
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_14/while/Identity"9
lstm_14_while_identitylstm_14/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
$__inference_signature_wrapper_499873
lstm_14_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_4987932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_14_input
?
?
while_cond_499192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_499192___redundant_placeholder04
0while_while_cond_499192___redundant_placeholder14
0while_while_cond_499192___redundant_placeholder24
0while_while_cond_499192___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?W
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_499712

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_499627*
condR
while_cond_499626*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::2
whilewhile:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
D__inference_dense_54_layer_call_and_return_conditional_losses_500887

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
~
)__inference_dense_54_layer_call_fn_500896

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_4997522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
while_cond_500288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_500288___redundant_placeholder04
0while_while_cond_500288___redundant_placeholder14
0while_while_cond_500288___redundant_placeholder24
0while_while_cond_500288___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?g
?
"__inference__traced_restore_501173
file_prefix$
 assignvariableop_dense_54_kernel$
 assignvariableop_1_dense_54_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate2
.assignvariableop_7_lstm_14_lstm_cell_14_kernel<
8assignvariableop_8_lstm_14_lstm_cell_14_recurrent_kernel0
,assignvariableop_9_lstm_14_lstm_cell_14_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1.
*assignvariableop_14_adam_dense_54_kernel_m,
(assignvariableop_15_adam_dense_54_bias_m:
6assignvariableop_16_adam_lstm_14_lstm_cell_14_kernel_mD
@assignvariableop_17_adam_lstm_14_lstm_cell_14_recurrent_kernel_m8
4assignvariableop_18_adam_lstm_14_lstm_cell_14_bias_m.
*assignvariableop_19_adam_dense_54_kernel_v,
(assignvariableop_20_adam_dense_54_bias_v:
6assignvariableop_21_adam_lstm_14_lstm_cell_14_kernel_vD
@assignvariableop_22_adam_lstm_14_lstm_cell_14_recurrent_kernel_v8
4assignvariableop_23_adam_lstm_14_lstm_cell_14_bias_v
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_54_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_54_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_14_lstm_cell_14_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_14_lstm_cell_14_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_14_lstm_cell_14_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_54_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_54_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_lstm_14_lstm_cell_14_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_lstm_14_lstm_cell_14_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_lstm_14_lstm_cell_14_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_54_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_54_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_lstm_14_lstm_cell_14_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_lstm_14_lstm_cell_14_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_14_lstm_cell_14_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24?
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_lstm_14_layer_call_fn_500538
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4992622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
'sequential_34_lstm_14_while_cond_498701H
Dsequential_34_lstm_14_while_sequential_34_lstm_14_while_loop_counterN
Jsequential_34_lstm_14_while_sequential_34_lstm_14_while_maximum_iterations+
'sequential_34_lstm_14_while_placeholder-
)sequential_34_lstm_14_while_placeholder_1-
)sequential_34_lstm_14_while_placeholder_2-
)sequential_34_lstm_14_while_placeholder_3J
Fsequential_34_lstm_14_while_less_sequential_34_lstm_14_strided_slice_1`
\sequential_34_lstm_14_while_sequential_34_lstm_14_while_cond_498701___redundant_placeholder0`
\sequential_34_lstm_14_while_sequential_34_lstm_14_while_cond_498701___redundant_placeholder1`
\sequential_34_lstm_14_while_sequential_34_lstm_14_while_cond_498701___redundant_placeholder2`
\sequential_34_lstm_14_while_sequential_34_lstm_14_while_cond_498701___redundant_placeholder3(
$sequential_34_lstm_14_while_identity
?
 sequential_34/lstm_14/while/LessLess'sequential_34_lstm_14_while_placeholderFsequential_34_lstm_14_while_less_sequential_34_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_34/lstm_14/while/Less?
$sequential_34/lstm_14/while/IdentityIdentity$sequential_34/lstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_34/lstm_14/while/Identity"U
$sequential_34_lstm_14_while_identity-sequential_34/lstm_14/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_500962

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????@:?????????@::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?9
?
while_body_499627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?W
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500374
inputs_0/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_500289*
condR
while_cond_500288*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?V
?
'sequential_34_lstm_14_while_body_498702H
Dsequential_34_lstm_14_while_sequential_34_lstm_14_while_loop_counterN
Jsequential_34_lstm_14_while_sequential_34_lstm_14_while_maximum_iterations+
'sequential_34_lstm_14_while_placeholder-
)sequential_34_lstm_14_while_placeholder_1-
)sequential_34_lstm_14_while_placeholder_2-
)sequential_34_lstm_14_while_placeholder_3G
Csequential_34_lstm_14_while_sequential_34_lstm_14_strided_slice_1_0?
sequential_34_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_34_lstm_14_tensorarrayunstack_tensorlistfromtensor_0M
Isequential_34_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0O
Ksequential_34_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0N
Jsequential_34_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0(
$sequential_34_lstm_14_while_identity*
&sequential_34_lstm_14_while_identity_1*
&sequential_34_lstm_14_while_identity_2*
&sequential_34_lstm_14_while_identity_3*
&sequential_34_lstm_14_while_identity_4*
&sequential_34_lstm_14_while_identity_5E
Asequential_34_lstm_14_while_sequential_34_lstm_14_strided_slice_1?
}sequential_34_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_34_lstm_14_tensorarrayunstack_tensorlistfromtensorK
Gsequential_34_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceM
Isequential_34_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceL
Hsequential_34_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource??
Msequential_34/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2O
Msequential_34/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
?sequential_34/lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_34_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_34_lstm_14_tensorarrayunstack_tensorlistfromtensor_0'sequential_34_lstm_14_while_placeholderVsequential_34/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02A
?sequential_34/lstm_14/while/TensorArrayV2Read/TensorListGetItem?
>sequential_34/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpIsequential_34_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02@
>sequential_34/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?
/sequential_34/lstm_14/while/lstm_cell_14/MatMulMatMulFsequential_34/lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_34/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_34/lstm_14/while/lstm_cell_14/MatMul?
@sequential_34/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpKsequential_34_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02B
@sequential_34/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
1sequential_34/lstm_14/while/lstm_cell_14/MatMul_1MatMul)sequential_34_lstm_14_while_placeholder_2Hsequential_34/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1sequential_34/lstm_14/while/lstm_cell_14/MatMul_1?
,sequential_34/lstm_14/while/lstm_cell_14/addAddV29sequential_34/lstm_14/while/lstm_cell_14/MatMul:product:0;sequential_34/lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_34/lstm_14/while/lstm_cell_14/add?
?sequential_34/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpJsequential_34_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02A
?sequential_34/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?
0sequential_34/lstm_14/while/lstm_cell_14/BiasAddBiasAdd0sequential_34/lstm_14/while/lstm_cell_14/add:z:0Gsequential_34/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_34/lstm_14/while/lstm_cell_14/BiasAdd?
.sequential_34/lstm_14/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_34/lstm_14/while/lstm_cell_14/Const?
8sequential_34/lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_34/lstm_14/while/lstm_cell_14/split/split_dim?
.sequential_34/lstm_14/while/lstm_cell_14/splitSplitAsequential_34/lstm_14/while/lstm_cell_14/split/split_dim:output:09sequential_34/lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split20
.sequential_34/lstm_14/while/lstm_cell_14/split?
0sequential_34/lstm_14/while/lstm_cell_14/SigmoidSigmoid7sequential_34/lstm_14/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@22
0sequential_34/lstm_14/while/lstm_cell_14/Sigmoid?
2sequential_34/lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid7sequential_34/lstm_14/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@24
2sequential_34/lstm_14/while/lstm_cell_14/Sigmoid_1?
,sequential_34/lstm_14/while/lstm_cell_14/mulMul6sequential_34/lstm_14/while/lstm_cell_14/Sigmoid_1:y:0)sequential_34_lstm_14_while_placeholder_3*
T0*'
_output_shapes
:?????????@2.
,sequential_34/lstm_14/while/lstm_cell_14/mul?
-sequential_34/lstm_14/while/lstm_cell_14/ReluRelu7sequential_34/lstm_14/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2/
-sequential_34/lstm_14/while/lstm_cell_14/Relu?
.sequential_34/lstm_14/while/lstm_cell_14/mul_1Mul4sequential_34/lstm_14/while/lstm_cell_14/Sigmoid:y:0;sequential_34/lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@20
.sequential_34/lstm_14/while/lstm_cell_14/mul_1?
.sequential_34/lstm_14/while/lstm_cell_14/add_1AddV20sequential_34/lstm_14/while/lstm_cell_14/mul:z:02sequential_34/lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@20
.sequential_34/lstm_14/while/lstm_cell_14/add_1?
2sequential_34/lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid7sequential_34/lstm_14/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@24
2sequential_34/lstm_14/while/lstm_cell_14/Sigmoid_2?
/sequential_34/lstm_14/while/lstm_cell_14/Relu_1Relu2sequential_34/lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@21
/sequential_34/lstm_14/while/lstm_cell_14/Relu_1?
.sequential_34/lstm_14/while/lstm_cell_14/mul_2Mul6sequential_34/lstm_14/while/lstm_cell_14/Sigmoid_2:y:0=sequential_34/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@20
.sequential_34/lstm_14/while/lstm_cell_14/mul_2?
@sequential_34/lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_34_lstm_14_while_placeholder_1'sequential_34_lstm_14_while_placeholder2sequential_34/lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_34/lstm_14/while/TensorArrayV2Write/TensorListSetItem?
!sequential_34/lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_34/lstm_14/while/add/y?
sequential_34/lstm_14/while/addAddV2'sequential_34_lstm_14_while_placeholder*sequential_34/lstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_34/lstm_14/while/add?
#sequential_34/lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_34/lstm_14/while/add_1/y?
!sequential_34/lstm_14/while/add_1AddV2Dsequential_34_lstm_14_while_sequential_34_lstm_14_while_loop_counter,sequential_34/lstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_34/lstm_14/while/add_1?
$sequential_34/lstm_14/while/IdentityIdentity%sequential_34/lstm_14/while/add_1:z:0*
T0*
_output_shapes
: 2&
$sequential_34/lstm_14/while/Identity?
&sequential_34/lstm_14/while/Identity_1IdentityJsequential_34_lstm_14_while_sequential_34_lstm_14_while_maximum_iterations*
T0*
_output_shapes
: 2(
&sequential_34/lstm_14/while/Identity_1?
&sequential_34/lstm_14/while/Identity_2Identity#sequential_34/lstm_14/while/add:z:0*
T0*
_output_shapes
: 2(
&sequential_34/lstm_14/while/Identity_2?
&sequential_34/lstm_14/while/Identity_3IdentityPsequential_34/lstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2(
&sequential_34/lstm_14/while/Identity_3?
&sequential_34/lstm_14/while/Identity_4Identity2sequential_34/lstm_14/while/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2(
&sequential_34/lstm_14/while/Identity_4?
&sequential_34/lstm_14/while/Identity_5Identity2sequential_34/lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2(
&sequential_34/lstm_14/while/Identity_5"U
$sequential_34_lstm_14_while_identity-sequential_34/lstm_14/while/Identity:output:0"Y
&sequential_34_lstm_14_while_identity_1/sequential_34/lstm_14/while/Identity_1:output:0"Y
&sequential_34_lstm_14_while_identity_2/sequential_34/lstm_14/while/Identity_2:output:0"Y
&sequential_34_lstm_14_while_identity_3/sequential_34/lstm_14/while/Identity_3:output:0"Y
&sequential_34_lstm_14_while_identity_4/sequential_34/lstm_14/while/Identity_4:output:0"Y
&sequential_34_lstm_14_while_identity_5/sequential_34/lstm_14/while/Identity_5:output:0"?
Hsequential_34_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resourceJsequential_34_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"?
Isequential_34_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceKsequential_34_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"?
Gsequential_34_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceIsequential_34_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"?
Asequential_34_lstm_14_while_sequential_34_lstm_14_strided_slice_1Csequential_34_lstm_14_while_sequential_34_lstm_14_strided_slice_1_0"?
}sequential_34_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_34_lstm_14_tensorarrayunstack_tensorlistfromtensorsequential_34_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_34_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?%
?
while_body_499325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_14_499349_0
while_lstm_cell_14_499351_0
while_lstm_cell_14_499353_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_14_499349
while_lstm_cell_14_499351
while_lstm_cell_14_499353??*while/lstm_cell_14/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_499349_0while_lstm_cell_14_499351_0while_lstm_cell_14_499353_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_4988992,
*while/lstm_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_14_499349while_lstm_cell_14_499349_0"8
while_lstm_cell_14_499351while_lstm_cell_14_499351_0"8
while_lstm_cell_14_499353while_lstm_cell_14_499353_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
??
?
!__inference__wrapped_model_498793
lstm_14_inputE
Asequential_34_lstm_14_lstm_cell_14_matmul_readvariableop_resourceG
Csequential_34_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceF
Bsequential_34_lstm_14_lstm_cell_14_biasadd_readvariableop_resource9
5sequential_34_dense_54_matmul_readvariableop_resource:
6sequential_34_dense_54_biasadd_readvariableop_resource
identity??sequential_34/lstm_14/whilew
sequential_34/lstm_14/ShapeShapelstm_14_input*
T0*
_output_shapes
:2
sequential_34/lstm_14/Shape?
)sequential_34/lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_34/lstm_14/strided_slice/stack?
+sequential_34/lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_34/lstm_14/strided_slice/stack_1?
+sequential_34/lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_34/lstm_14/strided_slice/stack_2?
#sequential_34/lstm_14/strided_sliceStridedSlice$sequential_34/lstm_14/Shape:output:02sequential_34/lstm_14/strided_slice/stack:output:04sequential_34/lstm_14/strided_slice/stack_1:output:04sequential_34/lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_34/lstm_14/strided_slice?
!sequential_34/lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_34/lstm_14/zeros/mul/y?
sequential_34/lstm_14/zeros/mulMul,sequential_34/lstm_14/strided_slice:output:0*sequential_34/lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_34/lstm_14/zeros/mul?
"sequential_34/lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_34/lstm_14/zeros/Less/y?
 sequential_34/lstm_14/zeros/LessLess#sequential_34/lstm_14/zeros/mul:z:0+sequential_34/lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_34/lstm_14/zeros/Less?
$sequential_34/lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_34/lstm_14/zeros/packed/1?
"sequential_34/lstm_14/zeros/packedPack,sequential_34/lstm_14/strided_slice:output:0-sequential_34/lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_34/lstm_14/zeros/packed?
!sequential_34/lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_34/lstm_14/zeros/Const?
sequential_34/lstm_14/zerosFill+sequential_34/lstm_14/zeros/packed:output:0*sequential_34/lstm_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_34/lstm_14/zeros?
#sequential_34/lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_34/lstm_14/zeros_1/mul/y?
!sequential_34/lstm_14/zeros_1/mulMul,sequential_34/lstm_14/strided_slice:output:0,sequential_34/lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_34/lstm_14/zeros_1/mul?
$sequential_34/lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_34/lstm_14/zeros_1/Less/y?
"sequential_34/lstm_14/zeros_1/LessLess%sequential_34/lstm_14/zeros_1/mul:z:0-sequential_34/lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_34/lstm_14/zeros_1/Less?
&sequential_34/lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2(
&sequential_34/lstm_14/zeros_1/packed/1?
$sequential_34/lstm_14/zeros_1/packedPack,sequential_34/lstm_14/strided_slice:output:0/sequential_34/lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_34/lstm_14/zeros_1/packed?
#sequential_34/lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_34/lstm_14/zeros_1/Const?
sequential_34/lstm_14/zeros_1Fill-sequential_34/lstm_14/zeros_1/packed:output:0,sequential_34/lstm_14/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_34/lstm_14/zeros_1?
$sequential_34/lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_34/lstm_14/transpose/perm?
sequential_34/lstm_14/transpose	Transposelstm_14_input-sequential_34/lstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:0?????????2!
sequential_34/lstm_14/transpose?
sequential_34/lstm_14/Shape_1Shape#sequential_34/lstm_14/transpose:y:0*
T0*
_output_shapes
:2
sequential_34/lstm_14/Shape_1?
+sequential_34/lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_34/lstm_14/strided_slice_1/stack?
-sequential_34/lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_34/lstm_14/strided_slice_1/stack_1?
-sequential_34/lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_34/lstm_14/strided_slice_1/stack_2?
%sequential_34/lstm_14/strided_slice_1StridedSlice&sequential_34/lstm_14/Shape_1:output:04sequential_34/lstm_14/strided_slice_1/stack:output:06sequential_34/lstm_14/strided_slice_1/stack_1:output:06sequential_34/lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_34/lstm_14/strided_slice_1?
1sequential_34/lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_34/lstm_14/TensorArrayV2/element_shape?
#sequential_34/lstm_14/TensorArrayV2TensorListReserve:sequential_34/lstm_14/TensorArrayV2/element_shape:output:0.sequential_34/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_34/lstm_14/TensorArrayV2?
Ksequential_34/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_34/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
=sequential_34/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_34/lstm_14/transpose:y:0Tsequential_34/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_34/lstm_14/TensorArrayUnstack/TensorListFromTensor?
+sequential_34/lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_34/lstm_14/strided_slice_2/stack?
-sequential_34/lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_34/lstm_14/strided_slice_2/stack_1?
-sequential_34/lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_34/lstm_14/strided_slice_2/stack_2?
%sequential_34/lstm_14/strided_slice_2StridedSlice#sequential_34/lstm_14/transpose:y:04sequential_34/lstm_14/strided_slice_2/stack:output:06sequential_34/lstm_14/strided_slice_2/stack_1:output:06sequential_34/lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2'
%sequential_34/lstm_14/strided_slice_2?
8sequential_34/lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpAsequential_34_lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02:
8sequential_34/lstm_14/lstm_cell_14/MatMul/ReadVariableOp?
)sequential_34/lstm_14/lstm_cell_14/MatMulMatMul.sequential_34/lstm_14/strided_slice_2:output:0@sequential_34/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_34/lstm_14/lstm_cell_14/MatMul?
:sequential_34/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpCsequential_34_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02<
:sequential_34/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp?
+sequential_34/lstm_14/lstm_cell_14/MatMul_1MatMul$sequential_34/lstm_14/zeros:output:0Bsequential_34/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+sequential_34/lstm_14/lstm_cell_14/MatMul_1?
&sequential_34/lstm_14/lstm_cell_14/addAddV23sequential_34/lstm_14/lstm_cell_14/MatMul:product:05sequential_34/lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_34/lstm_14/lstm_cell_14/add?
9sequential_34/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpBsequential_34_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_34/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp?
*sequential_34/lstm_14/lstm_cell_14/BiasAddBiasAdd*sequential_34/lstm_14/lstm_cell_14/add:z:0Asequential_34/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_34/lstm_14/lstm_cell_14/BiasAdd?
(sequential_34/lstm_14/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_34/lstm_14/lstm_cell_14/Const?
2sequential_34/lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_34/lstm_14/lstm_cell_14/split/split_dim?
(sequential_34/lstm_14/lstm_cell_14/splitSplit;sequential_34/lstm_14/lstm_cell_14/split/split_dim:output:03sequential_34/lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2*
(sequential_34/lstm_14/lstm_cell_14/split?
*sequential_34/lstm_14/lstm_cell_14/SigmoidSigmoid1sequential_34/lstm_14/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2,
*sequential_34/lstm_14/lstm_cell_14/Sigmoid?
,sequential_34/lstm_14/lstm_cell_14/Sigmoid_1Sigmoid1sequential_34/lstm_14/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2.
,sequential_34/lstm_14/lstm_cell_14/Sigmoid_1?
&sequential_34/lstm_14/lstm_cell_14/mulMul0sequential_34/lstm_14/lstm_cell_14/Sigmoid_1:y:0&sequential_34/lstm_14/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2(
&sequential_34/lstm_14/lstm_cell_14/mul?
'sequential_34/lstm_14/lstm_cell_14/ReluRelu1sequential_34/lstm_14/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2)
'sequential_34/lstm_14/lstm_cell_14/Relu?
(sequential_34/lstm_14/lstm_cell_14/mul_1Mul.sequential_34/lstm_14/lstm_cell_14/Sigmoid:y:05sequential_34/lstm_14/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2*
(sequential_34/lstm_14/lstm_cell_14/mul_1?
(sequential_34/lstm_14/lstm_cell_14/add_1AddV2*sequential_34/lstm_14/lstm_cell_14/mul:z:0,sequential_34/lstm_14/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2*
(sequential_34/lstm_14/lstm_cell_14/add_1?
,sequential_34/lstm_14/lstm_cell_14/Sigmoid_2Sigmoid1sequential_34/lstm_14/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2.
,sequential_34/lstm_14/lstm_cell_14/Sigmoid_2?
)sequential_34/lstm_14/lstm_cell_14/Relu_1Relu,sequential_34/lstm_14/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2+
)sequential_34/lstm_14/lstm_cell_14/Relu_1?
(sequential_34/lstm_14/lstm_cell_14/mul_2Mul0sequential_34/lstm_14/lstm_cell_14/Sigmoid_2:y:07sequential_34/lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2*
(sequential_34/lstm_14/lstm_cell_14/mul_2?
3sequential_34/lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   25
3sequential_34/lstm_14/TensorArrayV2_1/element_shape?
%sequential_34/lstm_14/TensorArrayV2_1TensorListReserve<sequential_34/lstm_14/TensorArrayV2_1/element_shape:output:0.sequential_34/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_34/lstm_14/TensorArrayV2_1z
sequential_34/lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_34/lstm_14/time?
.sequential_34/lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential_34/lstm_14/while/maximum_iterations?
(sequential_34/lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_34/lstm_14/while/loop_counter?
sequential_34/lstm_14/whileWhile1sequential_34/lstm_14/while/loop_counter:output:07sequential_34/lstm_14/while/maximum_iterations:output:0#sequential_34/lstm_14/time:output:0.sequential_34/lstm_14/TensorArrayV2_1:handle:0$sequential_34/lstm_14/zeros:output:0&sequential_34/lstm_14/zeros_1:output:0.sequential_34/lstm_14/strided_slice_1:output:0Msequential_34/lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_34_lstm_14_lstm_cell_14_matmul_readvariableop_resourceCsequential_34_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceBsequential_34_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_34_lstm_14_while_body_498702*3
cond+R)
'sequential_34_lstm_14_while_cond_498701*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
sequential_34/lstm_14/while?
Fsequential_34/lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2H
Fsequential_34/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape?
8sequential_34/lstm_14/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_34/lstm_14/while:output:3Osequential_34/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02:
8sequential_34/lstm_14/TensorArrayV2Stack/TensorListStack?
+sequential_34/lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+sequential_34/lstm_14/strided_slice_3/stack?
-sequential_34/lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_34/lstm_14/strided_slice_3/stack_1?
-sequential_34/lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_34/lstm_14/strided_slice_3/stack_2?
%sequential_34/lstm_14/strided_slice_3StridedSliceAsequential_34/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:04sequential_34/lstm_14/strided_slice_3/stack:output:06sequential_34/lstm_14/strided_slice_3/stack_1:output:06sequential_34/lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2'
%sequential_34/lstm_14/strided_slice_3?
&sequential_34/lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_34/lstm_14/transpose_1/perm?
!sequential_34/lstm_14/transpose_1	TransposeAsequential_34/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_34/lstm_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2#
!sequential_34/lstm_14/transpose_1?
sequential_34/lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_34/lstm_14/runtime?
,sequential_34/dense_54/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_54_matmul_readvariableop_resource*
_output_shapes

:@H*
dtype02.
,sequential_34/dense_54/MatMul/ReadVariableOp?
sequential_34/dense_54/MatMulMatMul.sequential_34/lstm_14/strided_slice_3:output:04sequential_34/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
sequential_34/dense_54/MatMul?
-sequential_34/dense_54/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_54_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_34/dense_54/BiasAdd/ReadVariableOp?
sequential_34/dense_54/BiasAddBiasAdd'sequential_34/dense_54/MatMul:product:05sequential_34/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2 
sequential_34/dense_54/BiasAdd?
IdentityIdentity'sequential_34/dense_54/BiasAdd:output:0^sequential_34/lstm_14/while*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2:
sequential_34/lstm_14/whilesequential_34/lstm_14/while:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_14_input
?W
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500527
inputs_0/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_500442*
condR
while_cond_500441*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?W
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_499559

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_499474*
condR
while_cond_499473*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::2
whilewhile:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
(__inference_lstm_14_layer_call_fn_500877

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4997122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_499785
lstm_14_input
lstm_14_499772
lstm_14_499774
lstm_14_499776
dense_54_499779
dense_54_499781
identity?? dense_54/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_499772lstm_14_499774lstm_14_499776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4997122!
lstm_14/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_54_499779dense_54_499781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_4997522"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_54/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_14_input
?9
?
while_body_499474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?W
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500702

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_14/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_500617*
condR
while_cond_500616*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::2
whilewhile:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
while_cond_500616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_500616___redundant_placeholder04
0while_while_cond_500616___redundant_placeholder14
0while_while_cond_500616___redundant_placeholder24
0while_while_cond_500616___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?
?
D__inference_dense_54_layer_call_and_return_conditional_losses_499752

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?D
?
lstm_14_while_body_499941,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0A
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0@
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor=
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource?
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource>
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource??
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItem?
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp?
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_14/while/lstm_cell_14/MatMul?
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_14/while/lstm_cell_14/MatMul_1?
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_14/while/lstm_cell_14/add?
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp?
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_14/while/lstm_cell_14/BiasAdd?
 lstm_14/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_14/while/lstm_cell_14/Const?
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dim?
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_14/while/lstm_cell_14/split?
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_14/while/lstm_cell_14/Sigmoid?
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_1?
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_14/while/lstm_cell_14/mul?
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_14/while/lstm_cell_14/Relu?
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_1?
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/add_1?
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_14/while/lstm_cell_14/Sigmoid_2?
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_14/while/lstm_cell_14/Relu_1?
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_14/while/lstm_cell_14/mul_2?
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_14/while/TensorArrayV2Write/TensorListSetIteml
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add/y?
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/addp
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add_1/y?
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1v
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_14/while/Identity?
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1x
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2?
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3?
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_4?
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_14/while/Identity_5"9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"?
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_500769
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_500769___redundant_placeholder04
0while_while_cond_500769___redundant_placeholder14
0while_while_cond_500769___redundant_placeholder24
0while_while_cond_500769___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?D
?
C__inference_lstm_14_layer_call_and_return_conditional_losses_499394

inputs
lstm_cell_14_499312
lstm_cell_14_499314
lstm_cell_14_499316
identity??$lstm_cell_14/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_499312lstm_cell_14_499314lstm_cell_14_499316*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_4988992&
$lstm_cell_14/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_499312lstm_cell_14_499314lstm_cell_14_499316*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????@:?????????@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_499325*
condR
while_cond_499324*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_14/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_498866

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????@:?????????@::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?
?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_498899

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????@:?????????@::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates:OK
'
_output_shapes
:?????????@
 
_user_specified_namestates
?
?
-__inference_lstm_cell_14_layer_call_fn_500979

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????@:?????????@:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_4988662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????@:?????????@:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?
?
.__inference_sequential_34_layer_call_fn_500221

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_4998352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_499769
lstm_14_input
lstm_14_499735
lstm_14_499737
lstm_14_499739
dense_54_499763
dense_54_499765
identity?? dense_54/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_499735lstm_14_499737lstm_14_499739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4995592!
lstm_14/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_54_499763dense_54_499765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_4997522"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_54/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_14_input
?
?
I__inference_sequential_34_layer_call_and_return_conditional_losses_499804

inputs
lstm_14_499791
lstm_14_499793
lstm_14_499795
dense_54_499798
dense_54_499800
identity?? dense_54/StatefulPartitionedCall?lstm_14/StatefulPartitionedCall?
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_499791lstm_14_499793lstm_14_499795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4995592!
lstm_14/StatefulPartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0dense_54_499798dense_54_499800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_4997522"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_54/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
while_cond_500441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_500441___redundant_placeholder04
0while_while_cond_500441___redundant_placeholder14
0while_while_cond_500441___redundant_placeholder24
0while_while_cond_500441___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_500617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_lstm_14_layer_call_fn_500866

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_lstm_14_layer_call_and_return_conditional_losses_4995592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_500929

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????@2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????@:?????????@::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
states/1
?9
?
while_body_500770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_499473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_499473___redundant_placeholder04
0while_while_cond_499473___redundant_placeholder14
0while_while_cond_499473___redundant_placeholder24
0while_while_cond_499473___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????@:?????????@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????@:-)
'
_output_shapes
:?????????@:

_output_shapes
: :

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_14_input:
serving_default_lstm_14_input:0?????????0<
dense_540
StatefulPartitionedCall:0?????????Htensorflow/serving/predict:??
?"
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
M__call__
*N&call_and_return_all_conditional_losses
O_default_save_signature"? 
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_14_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_14", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_34", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_14_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_14", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [{"class_name": "MeanAbsoluteError", "config": {"name": "mean_absolute_error", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
_inbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_14", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 1]}}
?
_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
iter

beta_1

beta_2
	decay
learning_ratemCmDmEmFmGvHvIvJvKvL"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
	variables
 layer_metrics
!layer_regularization_losses

"layers
trainable_variables
regularization_losses
#non_trainable_variables
M__call__
O_default_save_signature
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
,
Tserving_default"
signature_map
?

kernel
recurrent_kernel
bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
U__call__
*V&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(metrics
	variables
)layer_metrics
*layer_regularization_losses

+layers

,states
trainable_variables
regularization_losses
-non_trainable_variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:@H2dense_54/kernel
:H2dense_54/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.metrics
	variables
trainable_variables
/layer_regularization_losses

0layers
1layer_metrics
regularization_losses
2non_trainable_variables
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	?2lstm_14/lstm_cell_14/kernel
8:6	@?2%lstm_14/lstm_cell_14/recurrent_kernel
(:&?2lstm_14/lstm_cell_14/bias
.
30
41"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5metrics
$	variables
%trainable_variables
6layer_regularization_losses

7layers
8layer_metrics
&regularization_losses
9non_trainable_variables
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	:total
	;count
<	variables
=	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	>total
	?count
@
_fn_kwargs
A	variables
B	keras_api"?
_tf_keras_metric?{"class_name": "MeanAbsoluteError", "name": "mean_absolute_error", "dtype": "float32", "config": {"name": "mean_absolute_error", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
:0
;1"
trackable_list_wrapper
-
<	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
&:$@H2Adam/dense_54/kernel/m
 :H2Adam/dense_54/bias/m
3:1	?2"Adam/lstm_14/lstm_cell_14/kernel/m
=:;	@?2,Adam/lstm_14/lstm_cell_14/recurrent_kernel/m
-:+?2 Adam/lstm_14/lstm_cell_14/bias/m
&:$@H2Adam/dense_54/kernel/v
 :H2Adam/dense_54/bias/v
3:1	?2"Adam/lstm_14/lstm_cell_14/kernel/v
=:;	@?2,Adam/lstm_14/lstm_cell_14/recurrent_kernel/v
-:+?2 Adam/lstm_14/lstm_cell_14/bias/v
?2?
.__inference_sequential_34_layer_call_fn_499817
.__inference_sequential_34_layer_call_fn_500206
.__inference_sequential_34_layer_call_fn_500221
.__inference_sequential_34_layer_call_fn_499848?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_34_layer_call_and_return_conditional_losses_500191
I__inference_sequential_34_layer_call_and_return_conditional_losses_499785
I__inference_sequential_34_layer_call_and_return_conditional_losses_499769
I__inference_sequential_34_layer_call_and_return_conditional_losses_500032?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_498793?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
lstm_14_input?????????0
?2?
(__inference_lstm_14_layer_call_fn_500877
(__inference_lstm_14_layer_call_fn_500538
(__inference_lstm_14_layer_call_fn_500866
(__inference_lstm_14_layer_call_fn_500549?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500527
C__inference_lstm_14_layer_call_and_return_conditional_losses_500374
C__inference_lstm_14_layer_call_and_return_conditional_losses_500855
C__inference_lstm_14_layer_call_and_return_conditional_losses_500702?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_54_layer_call_fn_500896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_54_layer_call_and_return_conditional_losses_500887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
9B7
$__inference_signature_wrapper_499873lstm_14_input
?2?
-__inference_lstm_cell_14_layer_call_fn_500979
-__inference_lstm_cell_14_layer_call_fn_500996?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_500929
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_500962?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_498793x:?7
0?-
+?(
lstm_14_input?????????0
? "3?0
.
dense_54"?
dense_54?????????H?
D__inference_dense_54_layer_call_and_return_conditional_losses_500887\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????H
? |
)__inference_dense_54_layer_call_fn_500896O/?,
%?"
 ?
inputs?????????@
? "??????????H?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500374}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????@
? ?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500527}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????@
? ?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500702m??<
5?2
$?!
inputs?????????0

 
p

 
? "%?"
?
0?????????@
? ?
C__inference_lstm_14_layer_call_and_return_conditional_losses_500855m??<
5?2
$?!
inputs?????????0

 
p 

 
? "%?"
?
0?????????@
? ?
(__inference_lstm_14_layer_call_fn_500538pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????@?
(__inference_lstm_14_layer_call_fn_500549pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????@?
(__inference_lstm_14_layer_call_fn_500866`??<
5?2
$?!
inputs?????????0

 
p

 
? "??????????@?
(__inference_lstm_14_layer_call_fn_500877`??<
5?2
$?!
inputs?????????0

 
p 

 
? "??????????@?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_500929???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
H__inference_lstm_cell_14_layer_call_and_return_conditional_losses_500962???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "s?p
i?f
?
0/0?????????@
E?B
?
0/1/0?????????@
?
0/1/1?????????@
? ?
-__inference_lstm_cell_14_layer_call_fn_500979???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
-__inference_lstm_cell_14_layer_call_fn_500996???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????@
"?
states/1?????????@
p 
? "c?`
?
0?????????@
A?>
?
1/0?????????@
?
1/1?????????@?
I__inference_sequential_34_layer_call_and_return_conditional_losses_499769rB??
8?5
+?(
lstm_14_input?????????0
p

 
? "%?"
?
0?????????H
? ?
I__inference_sequential_34_layer_call_and_return_conditional_losses_499785rB??
8?5
+?(
lstm_14_input?????????0
p 

 
? "%?"
?
0?????????H
? ?
I__inference_sequential_34_layer_call_and_return_conditional_losses_500032k;?8
1?.
$?!
inputs?????????0
p

 
? "%?"
?
0?????????H
? ?
I__inference_sequential_34_layer_call_and_return_conditional_losses_500191k;?8
1?.
$?!
inputs?????????0
p 

 
? "%?"
?
0?????????H
? ?
.__inference_sequential_34_layer_call_fn_499817eB??
8?5
+?(
lstm_14_input?????????0
p

 
? "??????????H?
.__inference_sequential_34_layer_call_fn_499848eB??
8?5
+?(
lstm_14_input?????????0
p 

 
? "??????????H?
.__inference_sequential_34_layer_call_fn_500206^;?8
1?.
$?!
inputs?????????0
p

 
? "??????????H?
.__inference_sequential_34_layer_call_fn_500221^;?8
1?.
$?!
inputs?????????0
p 

 
? "??????????H?
$__inference_signature_wrapper_499873?K?H
? 
A?>
<
lstm_14_input+?(
lstm_14_input?????????0"3?0
.
dense_54"?
dense_54?????????H