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
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_52/kernel
s
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
_output_shapes

:@*
dtype0
r
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_52/bias
k
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes
:*
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
lstm_12/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_12/lstm_cell_12/kernel
?
/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_12/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*6
shared_name'%lstm_12/lstm_cell_12/recurrent_kernel
?
9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_12/recurrent_kernel*
_output_shapes
:	@?*
dtype0
?
lstm_12/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_12/lstm_cell_12/bias
?
-lstm_12/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/bias*
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
Adam/dense_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_52/kernel/m
?
*Adam/dense_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_52/bias/m
y
(Adam/dense_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/m*
_output_shapes
:*
dtype0
?
"Adam/lstm_12/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/m
?
6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
?
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_12/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/m
?
4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_52/kernel/v
?
*Adam/dense_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_52/bias/v
y
(Adam/dense_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/v*
_output_shapes
:*
dtype0
?
"Adam/lstm_12/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/v
?
6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
?
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v*
_output_shapes
:	@?*
dtype0
?
 Adam/lstm_12/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/v
?
4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/v*
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
regularization_losses
trainable_variables
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
regularization_losses
trainable_variables
	keras_api
|
_inbound_nodes

kernel
bias
	variables
regularization_losses
trainable_variables
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
 
#
0
1
2
3
4
?
	variables

layers
 metrics
regularization_losses
!layer_metrics
"non_trainable_variables
#layer_regularization_losses
trainable_variables
 
~

kernel
recurrent_kernel
bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
 
 

0
1
2
 

0
1
2
?
	variables

(layers
)metrics
regularization_losses

*states
+layer_metrics
,non_trainable_variables
-layer_regularization_losses
trainable_variables
 
[Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_52/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables

.layers
/metrics
regularization_losses
0layer_metrics
1non_trainable_variables
2layer_regularization_losses
trainable_variables
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
VARIABLE_VALUElstm_12/lstm_cell_12/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_12/lstm_cell_12/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_12/lstm_cell_12/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1

30
41
 
 
 

0
1
2
 

0
1
2
?
$	variables

5layers
6metrics
%regularization_losses
7layer_metrics
8non_trainable_variables
9layer_regularization_losses
&trainable_variables

	0
 
 
 
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
VARIABLE_VALUEAdam/dense_52/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_52/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_52/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_52/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:?????????0*
dtype0* 
shape:?????????0
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biasdense_52/kerneldense_52/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_477065
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOp9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOp-lstm_12/lstm_cell_12/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_52/kernel/m/Read/ReadVariableOp(Adam/dense_52/bias/m/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOp*Adam/dense_52/kernel/v/Read/ReadVariableOp(Adam/dense_52/bias/v/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpConst*%
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
__inference__traced_save_478283
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_52/kerneldense_52/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biastotalcounttotal_1count_1Adam/dense_52/kernel/mAdam/dense_52/bias/m"Adam/lstm_12/lstm_cell_12/kernel/m,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m Adam/lstm_12/lstm_cell_12/bias/mAdam/dense_52/kernel/vAdam/dense_52/bias/v"Adam/lstm_12/lstm_cell_12/kernel/v,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v Adam/lstm_12/lstm_cell_12/bias/v*$
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
"__inference__traced_restore_478365??
?
?
$__inference_signature_wrapper_477065
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_4759852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_12_input
?D
?
lstm_12_while_body_477292,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0A
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0@
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor=
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource?
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource>
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource??
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItem?
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/MatMul?
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_12/while/lstm_cell_12/MatMul_1?
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/add?
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/BiasAdd?
 lstm_12/while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_12/while/lstm_cell_12/Const?
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim?
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_12/while/lstm_cell_12/split?
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_12/while/lstm_cell_12/Sigmoid?
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_1?
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_12/while/lstm_cell_12/mul?
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_12/while/lstm_cell_12/Relu?
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_1?
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/add_1?
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_2?
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_12/while/lstm_cell_12/Relu_1?
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_2?
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_12/while/TensorArrayV2Write/TensorListSetIteml
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add/y?
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/addp
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add_1/y?
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1v
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_12/while/Identity?
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1x
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2?
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3?
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_4?
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_5"9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*Q
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
?
I__inference_sequential_32_layer_call_and_return_conditional_losses_476961
lstm_12_input
lstm_12_476927
lstm_12_476929
lstm_12_476931
dense_52_476955
dense_52_476957
identity?? dense_52/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_476927lstm_12_476929lstm_12_476931*
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4767512!
lstm_12/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_52_476955dense_52_476957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_4769442"
 dense_52/StatefulPartitionedCall?
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_52/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_12_input
?
?
-__inference_lstm_cell_12_layer_call_fn_478188

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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_4760912
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
?
~
)__inference_dense_52_layer_call_fn_478088

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_4769442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
while_cond_477480
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_477480___redundant_placeholder04
0while_while_cond_477480___redundant_placeholder14
0while_while_cond_477480___redundant_placeholder24
0while_while_cond_477480___redundant_placeholder3
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
?
?
I__inference_sequential_32_layer_call_and_return_conditional_losses_476977
lstm_12_input
lstm_12_476964
lstm_12_476966
lstm_12_476968
dense_52_476971
dense_52_476973
identity?? dense_52/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_476964lstm_12_476966lstm_12_476968*
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4769042!
lstm_12/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_52_476971dense_52_476973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_4769442"
 dense_52/StatefulPartitionedCall?
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_52/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_12_input
?9
?
while_body_477634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
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
?

?
lstm_12_while_cond_477132,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1D
@lstm_12_while_lstm_12_while_cond_477132___redundant_placeholder0D
@lstm_12_while_lstm_12_while_cond_477132___redundant_placeholder1D
@lstm_12_while_lstm_12_while_cond_477132___redundant_placeholder2D
@lstm_12_while_lstm_12_while_cond_477132___redundant_placeholder3
lstm_12_while_identity
?
lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2
lstm_12/while/Lessu
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_12/while/Identity"9
lstm_12_while_identitylstm_12/while/Identity:output:0*S
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
while_body_477809
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
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
?9
?
while_body_477481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
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
?
?
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_478121

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
?
?
while_cond_476384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_476384___redundant_placeholder04
0while_while_cond_476384___redundant_placeholder14
0while_while_cond_476384___redundant_placeholder24
0while_while_cond_476384___redundant_placeholder3
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
?
?
while_cond_476818
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_476818___redundant_placeholder04
0while_while_cond_476818___redundant_placeholder14
0while_while_cond_476818___redundant_placeholder24
0while_while_cond_476818___redundant_placeholder3
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_476586

inputs
lstm_cell_12_476504
lstm_cell_12_476506
lstm_cell_12_476508
identity??$lstm_cell_12/StatefulPartitionedCall?whileD
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_476504lstm_cell_12_476506lstm_cell_12_476508*
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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_4760912&
$lstm_cell_12/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_476504lstm_cell_12_476506lstm_cell_12_476508*
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
while_body_476517*
condR
while_cond_476516*K
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
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_12/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_476665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_476665___redundant_placeholder04
0while_while_cond_476665___redundant_placeholder14
0while_while_cond_476665___redundant_placeholder24
0while_while_cond_476665___redundant_placeholder3
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_476454

inputs
lstm_cell_12_476372
lstm_cell_12_476374
lstm_cell_12_476376
identity??$lstm_cell_12/StatefulPartitionedCall?whileD
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_476372lstm_cell_12_476374lstm_cell_12_476376*
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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_4760582&
$lstm_cell_12/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_476372lstm_cell_12_476374lstm_cell_12_476376*
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
while_body_476385*
condR
while_cond_476384*K
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
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_12/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_32_layer_call_fn_477009
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_32_layer_call_and_return_conditional_losses_4769962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_12_input
?D
?
lstm_12_while_body_477133,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0A
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0@
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor=
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource?
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource>
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource??
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItem?
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/MatMul?
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_12/while/lstm_cell_12/MatMul_1?
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/add?
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/BiasAdd?
 lstm_12/while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_12/while/lstm_cell_12/Const?
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim?
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2"
 lstm_12/while/lstm_cell_12/split?
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2$
"lstm_12/while/lstm_cell_12/Sigmoid?
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_1?
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*'
_output_shapes
:?????????@2 
lstm_12/while/lstm_cell_12/mul?
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2!
lstm_12/while/lstm_cell_12/Relu?
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_1?
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/add_1?
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2&
$lstm_12/while/lstm_cell_12/Sigmoid_2?
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2#
!lstm_12/while/lstm_cell_12/Relu_1?
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2"
 lstm_12/while/lstm_cell_12/mul_2?
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_12/while/TensorArrayV2Write/TensorListSetIteml
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add/y?
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/addp
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add_1/y?
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1v
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_12/while/Identity?
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1x
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2?
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3?
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_4?
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/while/Identity_5"9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*Q
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
(__inference_lstm_12_layer_call_fn_478058

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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4767512
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
?%
?
while_body_476517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_12_476541_0
while_lstm_cell_12_476543_0
while_lstm_cell_12_476545_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_12_476541
while_lstm_cell_12_476543
while_lstm_cell_12_476545??*while/lstm_cell_12/StatefulPartitionedCall?
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_476541_0while_lstm_cell_12_476543_0while_lstm_cell_12_476545_0*
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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_4760912,
*while/lstm_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1+^while/lstm_cell_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2+^while/lstm_cell_12/StatefulPartitionedCall*
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
while_lstm_cell_12_476541while_lstm_cell_12_476541_0"8
while_lstm_cell_12_476543while_lstm_cell_12_476543_0"8
while_lstm_cell_12_476545while_lstm_cell_12_476545_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_477894

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_477809*
condR
while_cond_477808*K
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
D__inference_dense_52_layer_call_and_return_conditional_losses_476944

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_475985
lstm_12_inputE
Asequential_32_lstm_12_lstm_cell_12_matmul_readvariableop_resourceG
Csequential_32_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceF
Bsequential_32_lstm_12_lstm_cell_12_biasadd_readvariableop_resource9
5sequential_32_dense_52_matmul_readvariableop_resource:
6sequential_32_dense_52_biasadd_readvariableop_resource
identity??sequential_32/lstm_12/whilew
sequential_32/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:2
sequential_32/lstm_12/Shape?
)sequential_32/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_32/lstm_12/strided_slice/stack?
+sequential_32/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_32/lstm_12/strided_slice/stack_1?
+sequential_32/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_32/lstm_12/strided_slice/stack_2?
#sequential_32/lstm_12/strided_sliceStridedSlice$sequential_32/lstm_12/Shape:output:02sequential_32/lstm_12/strided_slice/stack:output:04sequential_32/lstm_12/strided_slice/stack_1:output:04sequential_32/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_32/lstm_12/strided_slice?
!sequential_32/lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_32/lstm_12/zeros/mul/y?
sequential_32/lstm_12/zeros/mulMul,sequential_32/lstm_12/strided_slice:output:0*sequential_32/lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_32/lstm_12/zeros/mul?
"sequential_32/lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_32/lstm_12/zeros/Less/y?
 sequential_32/lstm_12/zeros/LessLess#sequential_32/lstm_12/zeros/mul:z:0+sequential_32/lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_32/lstm_12/zeros/Less?
$sequential_32/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_32/lstm_12/zeros/packed/1?
"sequential_32/lstm_12/zeros/packedPack,sequential_32/lstm_12/strided_slice:output:0-sequential_32/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_32/lstm_12/zeros/packed?
!sequential_32/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_32/lstm_12/zeros/Const?
sequential_32/lstm_12/zerosFill+sequential_32/lstm_12/zeros/packed:output:0*sequential_32/lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_32/lstm_12/zeros?
#sequential_32/lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_32/lstm_12/zeros_1/mul/y?
!sequential_32/lstm_12/zeros_1/mulMul,sequential_32/lstm_12/strided_slice:output:0,sequential_32/lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_32/lstm_12/zeros_1/mul?
$sequential_32/lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_32/lstm_12/zeros_1/Less/y?
"sequential_32/lstm_12/zeros_1/LessLess%sequential_32/lstm_12/zeros_1/mul:z:0-sequential_32/lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_32/lstm_12/zeros_1/Less?
&sequential_32/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2(
&sequential_32/lstm_12/zeros_1/packed/1?
$sequential_32/lstm_12/zeros_1/packedPack,sequential_32/lstm_12/strided_slice:output:0/sequential_32/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_32/lstm_12/zeros_1/packed?
#sequential_32/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_32/lstm_12/zeros_1/Const?
sequential_32/lstm_12/zeros_1Fill-sequential_32/lstm_12/zeros_1/packed:output:0,sequential_32/lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
sequential_32/lstm_12/zeros_1?
$sequential_32/lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_32/lstm_12/transpose/perm?
sequential_32/lstm_12/transpose	Transposelstm_12_input-sequential_32/lstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:0?????????2!
sequential_32/lstm_12/transpose?
sequential_32/lstm_12/Shape_1Shape#sequential_32/lstm_12/transpose:y:0*
T0*
_output_shapes
:2
sequential_32/lstm_12/Shape_1?
+sequential_32/lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_32/lstm_12/strided_slice_1/stack?
-sequential_32/lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_32/lstm_12/strided_slice_1/stack_1?
-sequential_32/lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_32/lstm_12/strided_slice_1/stack_2?
%sequential_32/lstm_12/strided_slice_1StridedSlice&sequential_32/lstm_12/Shape_1:output:04sequential_32/lstm_12/strided_slice_1/stack:output:06sequential_32/lstm_12/strided_slice_1/stack_1:output:06sequential_32/lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_32/lstm_12/strided_slice_1?
1sequential_32/lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_32/lstm_12/TensorArrayV2/element_shape?
#sequential_32/lstm_12/TensorArrayV2TensorListReserve:sequential_32/lstm_12/TensorArrayV2/element_shape:output:0.sequential_32/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_32/lstm_12/TensorArrayV2?
Ksequential_32/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_32/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
=sequential_32/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_32/lstm_12/transpose:y:0Tsequential_32/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_32/lstm_12/TensorArrayUnstack/TensorListFromTensor?
+sequential_32/lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_32/lstm_12/strided_slice_2/stack?
-sequential_32/lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_32/lstm_12/strided_slice_2/stack_1?
-sequential_32/lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_32/lstm_12/strided_slice_2/stack_2?
%sequential_32/lstm_12/strided_slice_2StridedSlice#sequential_32/lstm_12/transpose:y:04sequential_32/lstm_12/strided_slice_2/stack:output:06sequential_32/lstm_12/strided_slice_2/stack_1:output:06sequential_32/lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2'
%sequential_32/lstm_12/strided_slice_2?
8sequential_32/lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpAsequential_32_lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02:
8sequential_32/lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
)sequential_32/lstm_12/lstm_cell_12/MatMulMatMul.sequential_32/lstm_12/strided_slice_2:output:0@sequential_32/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_32/lstm_12/lstm_cell_12/MatMul?
:sequential_32/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpCsequential_32_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02<
:sequential_32/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
+sequential_32/lstm_12/lstm_cell_12/MatMul_1MatMul$sequential_32/lstm_12/zeros:output:0Bsequential_32/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+sequential_32/lstm_12/lstm_cell_12/MatMul_1?
&sequential_32/lstm_12/lstm_cell_12/addAddV23sequential_32/lstm_12/lstm_cell_12/MatMul:product:05sequential_32/lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_32/lstm_12/lstm_cell_12/add?
9sequential_32/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpBsequential_32_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_32/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
*sequential_32/lstm_12/lstm_cell_12/BiasAddBiasAdd*sequential_32/lstm_12/lstm_cell_12/add:z:0Asequential_32/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_32/lstm_12/lstm_cell_12/BiasAdd?
(sequential_32/lstm_12/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_32/lstm_12/lstm_cell_12/Const?
2sequential_32/lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_32/lstm_12/lstm_cell_12/split/split_dim?
(sequential_32/lstm_12/lstm_cell_12/splitSplit;sequential_32/lstm_12/lstm_cell_12/split/split_dim:output:03sequential_32/lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2*
(sequential_32/lstm_12/lstm_cell_12/split?
*sequential_32/lstm_12/lstm_cell_12/SigmoidSigmoid1sequential_32/lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2,
*sequential_32/lstm_12/lstm_cell_12/Sigmoid?
,sequential_32/lstm_12/lstm_cell_12/Sigmoid_1Sigmoid1sequential_32/lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2.
,sequential_32/lstm_12/lstm_cell_12/Sigmoid_1?
&sequential_32/lstm_12/lstm_cell_12/mulMul0sequential_32/lstm_12/lstm_cell_12/Sigmoid_1:y:0&sequential_32/lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2(
&sequential_32/lstm_12/lstm_cell_12/mul?
'sequential_32/lstm_12/lstm_cell_12/ReluRelu1sequential_32/lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2)
'sequential_32/lstm_12/lstm_cell_12/Relu?
(sequential_32/lstm_12/lstm_cell_12/mul_1Mul.sequential_32/lstm_12/lstm_cell_12/Sigmoid:y:05sequential_32/lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2*
(sequential_32/lstm_12/lstm_cell_12/mul_1?
(sequential_32/lstm_12/lstm_cell_12/add_1AddV2*sequential_32/lstm_12/lstm_cell_12/mul:z:0,sequential_32/lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2*
(sequential_32/lstm_12/lstm_cell_12/add_1?
,sequential_32/lstm_12/lstm_cell_12/Sigmoid_2Sigmoid1sequential_32/lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2.
,sequential_32/lstm_12/lstm_cell_12/Sigmoid_2?
)sequential_32/lstm_12/lstm_cell_12/Relu_1Relu,sequential_32/lstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2+
)sequential_32/lstm_12/lstm_cell_12/Relu_1?
(sequential_32/lstm_12/lstm_cell_12/mul_2Mul0sequential_32/lstm_12/lstm_cell_12/Sigmoid_2:y:07sequential_32/lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2*
(sequential_32/lstm_12/lstm_cell_12/mul_2?
3sequential_32/lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   25
3sequential_32/lstm_12/TensorArrayV2_1/element_shape?
%sequential_32/lstm_12/TensorArrayV2_1TensorListReserve<sequential_32/lstm_12/TensorArrayV2_1/element_shape:output:0.sequential_32/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_32/lstm_12/TensorArrayV2_1z
sequential_32/lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_32/lstm_12/time?
.sequential_32/lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential_32/lstm_12/while/maximum_iterations?
(sequential_32/lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_32/lstm_12/while/loop_counter?
sequential_32/lstm_12/whileWhile1sequential_32/lstm_12/while/loop_counter:output:07sequential_32/lstm_12/while/maximum_iterations:output:0#sequential_32/lstm_12/time:output:0.sequential_32/lstm_12/TensorArrayV2_1:handle:0$sequential_32/lstm_12/zeros:output:0&sequential_32/lstm_12/zeros_1:output:0.sequential_32/lstm_12/strided_slice_1:output:0Msequential_32/lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_32_lstm_12_lstm_cell_12_matmul_readvariableop_resourceCsequential_32_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceBsequential_32_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
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
'sequential_32_lstm_12_while_body_475894*3
cond+R)
'sequential_32_lstm_12_while_cond_475893*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
sequential_32/lstm_12/while?
Fsequential_32/lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2H
Fsequential_32/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
8sequential_32/lstm_12/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_32/lstm_12/while:output:3Osequential_32/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02:
8sequential_32/lstm_12/TensorArrayV2Stack/TensorListStack?
+sequential_32/lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+sequential_32/lstm_12/strided_slice_3/stack?
-sequential_32/lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_32/lstm_12/strided_slice_3/stack_1?
-sequential_32/lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_32/lstm_12/strided_slice_3/stack_2?
%sequential_32/lstm_12/strided_slice_3StridedSliceAsequential_32/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:04sequential_32/lstm_12/strided_slice_3/stack:output:06sequential_32/lstm_12/strided_slice_3/stack_1:output:06sequential_32/lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2'
%sequential_32/lstm_12/strided_slice_3?
&sequential_32/lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_32/lstm_12/transpose_1/perm?
!sequential_32/lstm_12/transpose_1	TransposeAsequential_32/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_32/lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2#
!sequential_32/lstm_12/transpose_1?
sequential_32/lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_32/lstm_12/runtime?
,sequential_32/dense_52/MatMul/ReadVariableOpReadVariableOp5sequential_32_dense_52_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_32/dense_52/MatMul/ReadVariableOp?
sequential_32/dense_52/MatMulMatMul.sequential_32/lstm_12/strided_slice_3:output:04sequential_32/dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_32/dense_52/MatMul?
-sequential_32/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_32_dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_32/dense_52/BiasAdd/ReadVariableOp?
sequential_32/dense_52/BiasAddBiasAdd'sequential_32/dense_52/MatMul:product:05sequential_32/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_32/dense_52/BiasAdd?
IdentityIdentity'sequential_32/dense_52/BiasAdd:output:0^sequential_32/lstm_12/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2:
sequential_32/lstm_12/whilesequential_32/lstm_12/while:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_12_input
?W
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_478047

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_477962*
condR
while_cond_477961*K
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
?9
?
while_body_476819
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
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
?
?
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_478154

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
?
?
.__inference_sequential_32_layer_call_fn_477398

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
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_32_layer_call_and_return_conditional_losses_4769962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
.__inference_sequential_32_layer_call_fn_477413

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
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_32_layer_call_and_return_conditional_losses_4770272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?9
?
while_body_477962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
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
.__inference_sequential_32_layer_call_fn_477040
lstm_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_32_layer_call_and_return_conditional_losses_4770272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????0
'
_user_specified_namelstm_12_input
?V
?
'sequential_32_lstm_12_while_body_475894H
Dsequential_32_lstm_12_while_sequential_32_lstm_12_while_loop_counterN
Jsequential_32_lstm_12_while_sequential_32_lstm_12_while_maximum_iterations+
'sequential_32_lstm_12_while_placeholder-
)sequential_32_lstm_12_while_placeholder_1-
)sequential_32_lstm_12_while_placeholder_2-
)sequential_32_lstm_12_while_placeholder_3G
Csequential_32_lstm_12_while_sequential_32_lstm_12_strided_slice_1_0?
sequential_32_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_32_lstm_12_tensorarrayunstack_tensorlistfromtensor_0M
Isequential_32_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0O
Ksequential_32_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0N
Jsequential_32_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0(
$sequential_32_lstm_12_while_identity*
&sequential_32_lstm_12_while_identity_1*
&sequential_32_lstm_12_while_identity_2*
&sequential_32_lstm_12_while_identity_3*
&sequential_32_lstm_12_while_identity_4*
&sequential_32_lstm_12_while_identity_5E
Asequential_32_lstm_12_while_sequential_32_lstm_12_strided_slice_1?
}sequential_32_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_32_lstm_12_tensorarrayunstack_tensorlistfromtensorK
Gsequential_32_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceM
Isequential_32_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceL
Hsequential_32_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource??
Msequential_32/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2O
Msequential_32/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
?sequential_32/lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_32_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_32_lstm_12_tensorarrayunstack_tensorlistfromtensor_0'sequential_32_lstm_12_while_placeholderVsequential_32/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02A
?sequential_32/lstm_12/while/TensorArrayV2Read/TensorListGetItem?
>sequential_32/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpIsequential_32_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02@
>sequential_32/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
/sequential_32/lstm_12/while/lstm_cell_12/MatMulMatMulFsequential_32/lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_32/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_32/lstm_12/while/lstm_cell_12/MatMul?
@sequential_32/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpKsequential_32_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02B
@sequential_32/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
1sequential_32/lstm_12/while/lstm_cell_12/MatMul_1MatMul)sequential_32_lstm_12_while_placeholder_2Hsequential_32/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1sequential_32/lstm_12/while/lstm_cell_12/MatMul_1?
,sequential_32/lstm_12/while/lstm_cell_12/addAddV29sequential_32/lstm_12/while/lstm_cell_12/MatMul:product:0;sequential_32/lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_32/lstm_12/while/lstm_cell_12/add?
?sequential_32/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpJsequential_32_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02A
?sequential_32/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
0sequential_32/lstm_12/while/lstm_cell_12/BiasAddBiasAdd0sequential_32/lstm_12/while/lstm_cell_12/add:z:0Gsequential_32/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_32/lstm_12/while/lstm_cell_12/BiasAdd?
.sequential_32/lstm_12/while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential_32/lstm_12/while/lstm_cell_12/Const?
8sequential_32/lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_32/lstm_12/while/lstm_cell_12/split/split_dim?
.sequential_32/lstm_12/while/lstm_cell_12/splitSplitAsequential_32/lstm_12/while/lstm_cell_12/split/split_dim:output:09sequential_32/lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split20
.sequential_32/lstm_12/while/lstm_cell_12/split?
0sequential_32/lstm_12/while/lstm_cell_12/SigmoidSigmoid7sequential_32/lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@22
0sequential_32/lstm_12/while/lstm_cell_12/Sigmoid?
2sequential_32/lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid7sequential_32/lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@24
2sequential_32/lstm_12/while/lstm_cell_12/Sigmoid_1?
,sequential_32/lstm_12/while/lstm_cell_12/mulMul6sequential_32/lstm_12/while/lstm_cell_12/Sigmoid_1:y:0)sequential_32_lstm_12_while_placeholder_3*
T0*'
_output_shapes
:?????????@2.
,sequential_32/lstm_12/while/lstm_cell_12/mul?
-sequential_32/lstm_12/while/lstm_cell_12/ReluRelu7sequential_32/lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2/
-sequential_32/lstm_12/while/lstm_cell_12/Relu?
.sequential_32/lstm_12/while/lstm_cell_12/mul_1Mul4sequential_32/lstm_12/while/lstm_cell_12/Sigmoid:y:0;sequential_32/lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@20
.sequential_32/lstm_12/while/lstm_cell_12/mul_1?
.sequential_32/lstm_12/while/lstm_cell_12/add_1AddV20sequential_32/lstm_12/while/lstm_cell_12/mul:z:02sequential_32/lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@20
.sequential_32/lstm_12/while/lstm_cell_12/add_1?
2sequential_32/lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid7sequential_32/lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@24
2sequential_32/lstm_12/while/lstm_cell_12/Sigmoid_2?
/sequential_32/lstm_12/while/lstm_cell_12/Relu_1Relu2sequential_32/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@21
/sequential_32/lstm_12/while/lstm_cell_12/Relu_1?
.sequential_32/lstm_12/while/lstm_cell_12/mul_2Mul6sequential_32/lstm_12/while/lstm_cell_12/Sigmoid_2:y:0=sequential_32/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@20
.sequential_32/lstm_12/while/lstm_cell_12/mul_2?
@sequential_32/lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_32_lstm_12_while_placeholder_1'sequential_32_lstm_12_while_placeholder2sequential_32/lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_32/lstm_12/while/TensorArrayV2Write/TensorListSetItem?
!sequential_32/lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_32/lstm_12/while/add/y?
sequential_32/lstm_12/while/addAddV2'sequential_32_lstm_12_while_placeholder*sequential_32/lstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_32/lstm_12/while/add?
#sequential_32/lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_32/lstm_12/while/add_1/y?
!sequential_32/lstm_12/while/add_1AddV2Dsequential_32_lstm_12_while_sequential_32_lstm_12_while_loop_counter,sequential_32/lstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_32/lstm_12/while/add_1?
$sequential_32/lstm_12/while/IdentityIdentity%sequential_32/lstm_12/while/add_1:z:0*
T0*
_output_shapes
: 2&
$sequential_32/lstm_12/while/Identity?
&sequential_32/lstm_12/while/Identity_1IdentityJsequential_32_lstm_12_while_sequential_32_lstm_12_while_maximum_iterations*
T0*
_output_shapes
: 2(
&sequential_32/lstm_12/while/Identity_1?
&sequential_32/lstm_12/while/Identity_2Identity#sequential_32/lstm_12/while/add:z:0*
T0*
_output_shapes
: 2(
&sequential_32/lstm_12/while/Identity_2?
&sequential_32/lstm_12/while/Identity_3IdentityPsequential_32/lstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2(
&sequential_32/lstm_12/while/Identity_3?
&sequential_32/lstm_12/while/Identity_4Identity2sequential_32/lstm_12/while/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2(
&sequential_32/lstm_12/while/Identity_4?
&sequential_32/lstm_12/while/Identity_5Identity2sequential_32/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2(
&sequential_32/lstm_12/while/Identity_5"U
$sequential_32_lstm_12_while_identity-sequential_32/lstm_12/while/Identity:output:0"Y
&sequential_32_lstm_12_while_identity_1/sequential_32/lstm_12/while/Identity_1:output:0"Y
&sequential_32_lstm_12_while_identity_2/sequential_32/lstm_12/while/Identity_2:output:0"Y
&sequential_32_lstm_12_while_identity_3/sequential_32/lstm_12/while/Identity_3:output:0"Y
&sequential_32_lstm_12_while_identity_4/sequential_32/lstm_12/while/Identity_4:output:0"Y
&sequential_32_lstm_12_while_identity_5/sequential_32/lstm_12/while/Identity_5:output:0"?
Hsequential_32_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resourceJsequential_32_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"?
Isequential_32_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceKsequential_32_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"?
Gsequential_32_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceIsequential_32_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
Asequential_32_lstm_12_while_sequential_32_lstm_12_strided_slice_1Csequential_32_lstm_12_while_sequential_32_lstm_12_strided_slice_1_0"?
}sequential_32_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_32_lstm_12_tensorarrayunstack_tensorlistfromtensorsequential_32_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_32_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*Q
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
while_cond_476516
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_476516___redundant_placeholder04
0while_while_cond_476516___redundant_placeholder14
0while_while_cond_476516___redundant_placeholder24
0while_while_cond_476516___redundant_placeholder3
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
?
?
while_cond_477633
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_477633___redundant_placeholder04
0while_while_cond_477633___redundant_placeholder14
0while_while_cond_477633___redundant_placeholder24
0while_while_cond_477633___redundant_placeholder3
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
?l
?
I__inference_sequential_32_layer_call_and_return_conditional_losses_477383

inputs7
3lstm_12_lstm_cell_12_matmul_readvariableop_resource9
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource8
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource+
'dense_52_matmul_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource
identity??lstm_12/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicel
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lessr
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zerosp
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessv
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zeros_1?
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm?
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:0?????????2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1?
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack?
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1?
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2?
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1?
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_12/TensorArrayV2/element_shape?
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor?
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack?
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1?
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2?
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_12/strided_slice_2?
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul?
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul_1?
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add?
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/BiasAddz
lstm_12/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/lstm_cell_12/Const?
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim?
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_12/lstm_cell_12/split?
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Sigmoid?
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_1?
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul?
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu?
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_1?
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/add_1?
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_2?
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu_1?
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_2?
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_12/TensorArrayV2_1/element_shape?
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2_1^
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/time?
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter?
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
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
lstm_12_while_body_477292*%
condR
lstm_12_while_cond_477291*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_12/while?
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack?
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_12/strided_slice_3/stack?
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1?
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2?
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_12/strided_slice_3?
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm?
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtime?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_52/MatMul/ReadVariableOp?
dense_52/MatMulMatMul lstm_12/strided_slice_3:output:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_52/MatMul?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_52/BiasAdd}
IdentityIdentitydense_52/BiasAdd:output:0^lstm_12/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2
lstm_12/whilelstm_12/while:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?W
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_476751

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_476666*
condR
while_cond_476665*K
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
(__inference_lstm_12_layer_call_fn_478069

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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4769042
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
?W
?
C__inference_lstm_12_layer_call_and_return_conditional_losses_476904

inputs/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_476819*
condR
while_cond_476818*K
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
(__inference_lstm_12_layer_call_fn_477741
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4765862
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
?
?
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_476091

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
?
?
D__inference_dense_52_layer_call_and_return_conditional_losses_478079

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?g
?
"__inference__traced_restore_478365
file_prefix$
 assignvariableop_dense_52_kernel$
 assignvariableop_1_dense_52_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate2
.assignvariableop_7_lstm_12_lstm_cell_12_kernel<
8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernel0
,assignvariableop_9_lstm_12_lstm_cell_12_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1.
*assignvariableop_14_adam_dense_52_kernel_m,
(assignvariableop_15_adam_dense_52_bias_m:
6assignvariableop_16_adam_lstm_12_lstm_cell_12_kernel_mD
@assignvariableop_17_adam_lstm_12_lstm_cell_12_recurrent_kernel_m8
4assignvariableop_18_adam_lstm_12_lstm_cell_12_bias_m.
*assignvariableop_19_adam_dense_52_kernel_v,
(assignvariableop_20_adam_dense_52_bias_v:
6assignvariableop_21_adam_lstm_12_lstm_cell_12_kernel_vD
@assignvariableop_22_adam_lstm_12_lstm_cell_12_recurrent_kernel_v8
4assignvariableop_23_adam_lstm_12_lstm_cell_12_bias_v
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
AssignVariableOpAssignVariableOp assignvariableop_dense_52_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_52_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_12_lstm_cell_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_12_lstm_cell_12_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_52_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_52_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_lstm_12_lstm_cell_12_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_lstm_12_lstm_cell_12_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_lstm_12_lstm_cell_12_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_52_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_52_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_lstm_12_lstm_cell_12_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_lstm_12_lstm_cell_12_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_12_lstm_cell_12_bias_vIdentity_23:output:0"/device:CPU:0*
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
(__inference_lstm_12_layer_call_fn_477730
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4764542
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
?
?
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_476058

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
?%
?
while_body_476385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_12_476409_0
while_lstm_cell_12_476411_0
while_lstm_cell_12_476413_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_12_476409
while_lstm_cell_12_476411
while_lstm_cell_12_476413??*while/lstm_cell_12/StatefulPartitionedCall?
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_476409_0while_lstm_cell_12_476411_0while_lstm_cell_12_476413_0*
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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_4760582,
*while/lstm_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_12/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1+^while/lstm_cell_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2+^while/lstm_cell_12/StatefulPartitionedCall*
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
while_lstm_cell_12_476409while_lstm_cell_12_476409_0"8
while_lstm_cell_12_476411while_lstm_cell_12_476411_0"8
while_lstm_cell_12_476413while_lstm_cell_12_476413_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????@:?????????@: : :::2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 
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
?

__inference__traced_save_478283
file_prefix.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableopD
@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop8
4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_52_kernel_m_read_readvariableop3
/savev2_adam_dense_52_bias_m_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop5
1savev2_adam_dense_52_kernel_v_read_readvariableop3
/savev2_adam_dense_52_bias_v_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableop
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
value3B1 B+_temp_4c8e369970c543a1b4494e9942d1a248/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_52_kernel_m_read_readvariableop/savev2_adam_dense_52_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop1savev2_adam_dense_52_kernel_v_read_readvariableop/savev2_adam_dense_52_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :@:: : : : : :	?:	@?:?: : : : :@::	?:	@?:?:@::	?:	@?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::
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

:@: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	@?:!

_output_shapes	
:?:$ 

_output_shapes

:@: 

_output_shapes
::%!

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
?
?
while_cond_477961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_477961___redundant_placeholder04
0while_while_cond_477961___redundant_placeholder14
0while_while_cond_477961___redundant_placeholder24
0while_while_cond_477961___redundant_placeholder3
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_477719
inputs_0/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_477634*
condR
while_cond_477633*K
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
?
?
I__inference_sequential_32_layer_call_and_return_conditional_losses_476996

inputs
lstm_12_476983
lstm_12_476985
lstm_12_476987
dense_52_476990
dense_52_476992
identity?? dense_52/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_476983lstm_12_476985lstm_12_476987*
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4767512!
lstm_12/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_52_476990dense_52_476992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_4769442"
 dense_52/StatefulPartitionedCall?
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_52/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?9
?
while_body_476666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_12_matmul_readvariableop_resource_09
5while_lstm_cell_12_matmul_1_readvariableop_resource_08
4while_lstm_cell_12_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_12_matmul_readvariableop_resource7
3while_lstm_cell_12_matmul_1_readvariableop_resource6
2while_lstm_cell_12_biasadd_readvariableop_resource??
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	@?*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAddv
while/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_12/Const?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0*
T0*'
_output_shapes
:?????????@2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0*
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
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
?
?
I__inference_sequential_32_layer_call_and_return_conditional_losses_477027

inputs
lstm_12_477014
lstm_12_477016
lstm_12_477018
dense_52_477021
dense_52_477023
identity?? dense_52/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_477014lstm_12_477016lstm_12_477018*
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_4769042!
lstm_12/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0dense_52_477021dense_52_477023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_4769442"
 dense_52/StatefulPartitionedCall?
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_52/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?l
?
I__inference_sequential_32_layer_call_and_return_conditional_losses_477224

inputs7
3lstm_12_lstm_cell_12_matmul_readvariableop_resource9
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource8
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource+
'dense_52_matmul_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource
identity??lstm_12/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicel
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lessr
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zerosp
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessv
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/zeros_1?
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm?
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:0?????????2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1?
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack?
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1?
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2?
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1?
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_12/TensorArrayV2/element_shape?
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor?
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack?
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1?
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2?
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_12/strided_slice_2?
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul?
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul_1?
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add?
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/BiasAddz
lstm_12/lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/lstm_cell_12/Const?
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim?
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_12/lstm_cell_12/split?
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Sigmoid?
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_1?
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul?
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu?
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_1?
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/add_1?
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2 
lstm_12/lstm_cell_12/Sigmoid_2?
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/Relu_1?
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_12/lstm_cell_12/mul_2?
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2'
%lstm_12/TensorArrayV2_1/element_shape?
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2_1^
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/time?
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter?
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
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
lstm_12_while_body_477133*%
condR
lstm_12_while_cond_477132*K
output_shapes:
8: : : : :?????????@:?????????@: : : : : *
parallel_iterations 2
lstm_12/while?
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:0?????????@*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack?
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_12/strided_slice_3/stack?
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1?
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2?
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask2
lstm_12/strided_slice_3?
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm?
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0@2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtime?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_52/MatMul/ReadVariableOp?
dense_52/MatMulMatMul lstm_12/strided_slice_3:output:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_52/MatMul?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_52/BiasAdd}
IdentityIdentitydense_52/BiasAdd:output:0^lstm_12/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0:::::2
lstm_12/whilelstm_12/while:S O
+
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
'sequential_32_lstm_12_while_cond_475893H
Dsequential_32_lstm_12_while_sequential_32_lstm_12_while_loop_counterN
Jsequential_32_lstm_12_while_sequential_32_lstm_12_while_maximum_iterations+
'sequential_32_lstm_12_while_placeholder-
)sequential_32_lstm_12_while_placeholder_1-
)sequential_32_lstm_12_while_placeholder_2-
)sequential_32_lstm_12_while_placeholder_3J
Fsequential_32_lstm_12_while_less_sequential_32_lstm_12_strided_slice_1`
\sequential_32_lstm_12_while_sequential_32_lstm_12_while_cond_475893___redundant_placeholder0`
\sequential_32_lstm_12_while_sequential_32_lstm_12_while_cond_475893___redundant_placeholder1`
\sequential_32_lstm_12_while_sequential_32_lstm_12_while_cond_475893___redundant_placeholder2`
\sequential_32_lstm_12_while_sequential_32_lstm_12_while_cond_475893___redundant_placeholder3(
$sequential_32_lstm_12_while_identity
?
 sequential_32/lstm_12/while/LessLess'sequential_32_lstm_12_while_placeholderFsequential_32_lstm_12_while_less_sequential_32_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_32/lstm_12/while/Less?
$sequential_32/lstm_12/while/IdentityIdentity$sequential_32/lstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_32/lstm_12/while/Identity"U
$sequential_32_lstm_12_while_identity-sequential_32/lstm_12/while/Identity:output:0*S
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
?
?
while_cond_477808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_477808___redundant_placeholder04
0while_while_cond_477808___redundant_placeholder14
0while_while_cond_477808___redundant_placeholder24
0while_while_cond_477808___redundant_placeholder3
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_477566
inputs_0/
+lstm_cell_12_matmul_readvariableop_resource1
-lstm_cell_12_matmul_1_readvariableop_resource0
,lstm_cell_12_biasadd_readvariableop_resource
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAddj
lstm_cell_12/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/Const~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????@:?????????@:?????????@:?????????@*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_477481*
condR
while_cond_477480*K
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
?

?
lstm_12_while_cond_477291,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1D
@lstm_12_while_lstm_12_while_cond_477291___redundant_placeholder0D
@lstm_12_while_lstm_12_while_cond_477291___redundant_placeholder1D
@lstm_12_while_lstm_12_while_cond_477291___redundant_placeholder2D
@lstm_12_while_lstm_12_while_cond_477291___redundant_placeholder3
lstm_12_while_identity
?
lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2
lstm_12/while/Lessu
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_12/while/Identity"9
lstm_12_while_identitylstm_12/while/Identity:output:0*S
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
?
?
-__inference_lstm_cell_12_layer_call_fn_478171

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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_4760582
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
states/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_12_input:
serving_default_lstm_12_input:0?????????0<
dense_520
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?"
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
M__call__
*N&call_and_return_all_conditional_losses
O_default_save_signature"? 
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_32", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_32", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_12_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_12", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 24, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_32", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_12_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_12", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 24, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [{"class_name": "MeanAbsoluteError", "config": {"name": "mean_absolute_error", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_12", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 1]}}
?
_inbound_nodes

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 24, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
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
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
	variables

layers
 metrics
regularization_losses
!layer_metrics
"non_trainable_variables
#layer_regularization_losses
trainable_variables
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
%regularization_losses
&trainable_variables
'	keras_api
U__call__
*V&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
	variables

(layers
)metrics
regularization_losses

*states
+layer_metrics
,non_trainable_variables
-layer_regularization_losses
trainable_variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
!:@2dense_52/kernel
:2dense_52/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables

.layers
/metrics
regularization_losses
0layer_metrics
1non_trainable_variables
2layer_regularization_losses
trainable_variables
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
.:,	?2lstm_12/lstm_cell_12/kernel
8:6	@?2%lstm_12/lstm_cell_12/recurrent_kernel
(:&?2lstm_12/lstm_cell_12/bias
.
0
1"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
$	variables

5layers
6metrics
%regularization_losses
7layer_metrics
8non_trainable_variables
9layer_regularization_losses
&trainable_variables
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
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
&:$@2Adam/dense_52/kernel/m
 :2Adam/dense_52/bias/m
3:1	?2"Adam/lstm_12/lstm_cell_12/kernel/m
=:;	@?2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
-:+?2 Adam/lstm_12/lstm_cell_12/bias/m
&:$@2Adam/dense_52/kernel/v
 :2Adam/dense_52/bias/v
3:1	?2"Adam/lstm_12/lstm_cell_12/kernel/v
=:;	@?2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
-:+?2 Adam/lstm_12/lstm_cell_12/bias/v
?2?
.__inference_sequential_32_layer_call_fn_477009
.__inference_sequential_32_layer_call_fn_477040
.__inference_sequential_32_layer_call_fn_477413
.__inference_sequential_32_layer_call_fn_477398?
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
I__inference_sequential_32_layer_call_and_return_conditional_losses_477383
I__inference_sequential_32_layer_call_and_return_conditional_losses_476961
I__inference_sequential_32_layer_call_and_return_conditional_losses_477224
I__inference_sequential_32_layer_call_and_return_conditional_losses_476977?
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
!__inference__wrapped_model_475985?
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
lstm_12_input?????????0
?2?
(__inference_lstm_12_layer_call_fn_478069
(__inference_lstm_12_layer_call_fn_477730
(__inference_lstm_12_layer_call_fn_477741
(__inference_lstm_12_layer_call_fn_478058?
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_477719
C__inference_lstm_12_layer_call_and_return_conditional_losses_477566
C__inference_lstm_12_layer_call_and_return_conditional_losses_478047
C__inference_lstm_12_layer_call_and_return_conditional_losses_477894?
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
)__inference_dense_52_layer_call_fn_478088?
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
D__inference_dense_52_layer_call_and_return_conditional_losses_478079?
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
$__inference_signature_wrapper_477065lstm_12_input
?2?
-__inference_lstm_cell_12_layer_call_fn_478188
-__inference_lstm_cell_12_layer_call_fn_478171?
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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_478121
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_478154?
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
!__inference__wrapped_model_475985x:?7
0?-
+?(
lstm_12_input?????????0
? "3?0
.
dense_52"?
dense_52??????????
D__inference_dense_52_layer_call_and_return_conditional_losses_478079\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? |
)__inference_dense_52_layer_call_fn_478088O/?,
%?"
 ?
inputs?????????@
? "???????????
C__inference_lstm_12_layer_call_and_return_conditional_losses_477566}O?L
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_477719}O?L
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_477894m??<
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
C__inference_lstm_12_layer_call_and_return_conditional_losses_478047m??<
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
(__inference_lstm_12_layer_call_fn_477730pO?L
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
(__inference_lstm_12_layer_call_fn_477741pO?L
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
(__inference_lstm_12_layer_call_fn_478058`??<
5?2
$?!
inputs?????????0

 
p

 
? "??????????@?
(__inference_lstm_12_layer_call_fn_478069`??<
5?2
$?!
inputs?????????0

 
p 

 
? "??????????@?
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_478121???}
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
H__inference_lstm_cell_12_layer_call_and_return_conditional_losses_478154???}
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
-__inference_lstm_cell_12_layer_call_fn_478171???}
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
-__inference_lstm_cell_12_layer_call_fn_478188???}
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
I__inference_sequential_32_layer_call_and_return_conditional_losses_476961rB??
8?5
+?(
lstm_12_input?????????0
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_32_layer_call_and_return_conditional_losses_476977rB??
8?5
+?(
lstm_12_input?????????0
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_32_layer_call_and_return_conditional_losses_477224k;?8
1?.
$?!
inputs?????????0
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_32_layer_call_and_return_conditional_losses_477383k;?8
1?.
$?!
inputs?????????0
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_32_layer_call_fn_477009eB??
8?5
+?(
lstm_12_input?????????0
p

 
? "???????????
.__inference_sequential_32_layer_call_fn_477040eB??
8?5
+?(
lstm_12_input?????????0
p 

 
? "???????????
.__inference_sequential_32_layer_call_fn_477398^;?8
1?.
$?!
inputs?????????0
p

 
? "???????????
.__inference_sequential_32_layer_call_fn_477413^;?8
1?.
$?!
inputs?????????0
p 

 
? "???????????
$__inference_signature_wrapper_477065?K?H
? 
A?>
<
lstm_12_input+?(
lstm_12_input?????????0"3?0
.
dense_52"?
dense_52?????????