╚■6
╛Н
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8вд0
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

Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:┘*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:┘*
dtype0

Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:┘*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:┘*
dtype0
И
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А┘*&
shared_nameAdam/v/dense_1/kernel
Б
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel* 
_output_shapes
:
А┘*
dtype0
И
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А┘*&
shared_nameAdam/m/dense_1/kernel
Б
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel* 
_output_shapes
:
А┘*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:А*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:А*
dtype0
Г
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
А*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	
А*
dtype0
Г
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
А*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	
А*
dtype0
А
Adam/v/conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_7/bias
y
(Adam/v/conv1d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_7/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_7/bias
y
(Adam/m/conv1d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_7/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_7/kernel
Е
*Adam/v/conv1d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_7/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_7/kernel
Е
*Adam/m/conv1d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_7/kernel*"
_output_shapes
:*
dtype0
А
Adam/v/conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_6/bias
y
(Adam/v/conv1d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_6/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_6/bias
y
(Adam/m/conv1d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_6/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_6/kernel
Е
*Adam/v/conv1d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_6/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_6/kernel
Е
*Adam/m/conv1d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_6/kernel*"
_output_shapes
:*
dtype0
Ъ
!Adam/v/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_7/beta
У
5Adam/v/layer_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_7/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_7/beta
У
5Adam/m/layer_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_7/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_7/gamma
Х
6Adam/v/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_7/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_7/gamma
Х
6Adam/m/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_7/gamma*
_output_shapes
:*
dtype0
в
%Adam/v/kf_attention_3/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/kf_attention_3/projection_bias
Ы
9Adam/v/kf_attention_3/projection_bias/Read/ReadVariableOpReadVariableOp%Adam/v/kf_attention_3/projection_bias*
_output_shapes
:*
dtype0
в
%Adam/m/kf_attention_3/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/kf_attention_3/projection_bias
Ы
9Adam/m/kf_attention_3/projection_bias/Read/ReadVariableOpReadVariableOp%Adam/m/kf_attention_3/projection_bias*
_output_shapes
:*
dtype0
п
'Adam/v/kf_attention_3/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adam/v/kf_attention_3/projection_kernel
и
;Adam/v/kf_attention_3/projection_kernel/Read/ReadVariableOpReadVariableOp'Adam/v/kf_attention_3/projection_kernel*#
_output_shapes
:А*
dtype0
п
'Adam/m/kf_attention_3/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adam/m/kf_attention_3/projection_kernel
и
;Adam/m/kf_attention_3/projection_kernel/Read/ReadVariableOpReadVariableOp'Adam/m/kf_attention_3/projection_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/v/kf_attention_3/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/kf_attention_3/value_kernel
Ю
6Adam/v/kf_attention_3/value_kernel/Read/ReadVariableOpReadVariableOp"Adam/v/kf_attention_3/value_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/m/kf_attention_3/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/kf_attention_3/value_kernel
Ю
6Adam/m/kf_attention_3/value_kernel/Read/ReadVariableOpReadVariableOp"Adam/m/kf_attention_3/value_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/v/kf_attention_3/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/kf_attention_3/query_kernel
Ю
6Adam/v/kf_attention_3/query_kernel/Read/ReadVariableOpReadVariableOp"Adam/v/kf_attention_3/query_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/m/kf_attention_3/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/kf_attention_3/query_kernel
Ю
6Adam/m/kf_attention_3/query_kernel/Read/ReadVariableOpReadVariableOp"Adam/m/kf_attention_3/query_kernel*#
_output_shapes
:А*
dtype0
Ъ
!Adam/v/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_6/beta
У
5Adam/v/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_6/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_6/beta
У
5Adam/m/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_6/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_6/gamma
Х
6Adam/v/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_6/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_6/gamma
Х
6Adam/m/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_6/gamma*
_output_shapes
:*
dtype0
А
Adam/v/conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_5/bias
y
(Adam/v/conv1d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_5/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_5/bias
y
(Adam/m/conv1d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_5/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_5/kernel
Е
*Adam/v/conv1d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_5/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_5/kernel
Е
*Adam/m/conv1d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_5/kernel*"
_output_shapes
:*
dtype0
А
Adam/v/conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_4/bias
y
(Adam/v/conv1d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_4/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_4/bias
y
(Adam/m/conv1d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_4/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_4/kernel
Е
*Adam/v/conv1d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_4/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_4/kernel
Е
*Adam/m/conv1d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_4/kernel*"
_output_shapes
:*
dtype0
Ъ
!Adam/v/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_5/beta
У
5Adam/v/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_5/beta
У
5Adam/m/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_5/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_5/gamma
Х
6Adam/v/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_5/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_5/gamma
Х
6Adam/m/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_5/gamma*
_output_shapes
:*
dtype0
в
%Adam/v/kf_attention_2/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/kf_attention_2/projection_bias
Ы
9Adam/v/kf_attention_2/projection_bias/Read/ReadVariableOpReadVariableOp%Adam/v/kf_attention_2/projection_bias*
_output_shapes
:*
dtype0
в
%Adam/m/kf_attention_2/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/kf_attention_2/projection_bias
Ы
9Adam/m/kf_attention_2/projection_bias/Read/ReadVariableOpReadVariableOp%Adam/m/kf_attention_2/projection_bias*
_output_shapes
:*
dtype0
п
'Adam/v/kf_attention_2/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adam/v/kf_attention_2/projection_kernel
и
;Adam/v/kf_attention_2/projection_kernel/Read/ReadVariableOpReadVariableOp'Adam/v/kf_attention_2/projection_kernel*#
_output_shapes
:А*
dtype0
п
'Adam/m/kf_attention_2/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adam/m/kf_attention_2/projection_kernel
и
;Adam/m/kf_attention_2/projection_kernel/Read/ReadVariableOpReadVariableOp'Adam/m/kf_attention_2/projection_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/v/kf_attention_2/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/kf_attention_2/value_kernel
Ю
6Adam/v/kf_attention_2/value_kernel/Read/ReadVariableOpReadVariableOp"Adam/v/kf_attention_2/value_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/m/kf_attention_2/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/kf_attention_2/value_kernel
Ю
6Adam/m/kf_attention_2/value_kernel/Read/ReadVariableOpReadVariableOp"Adam/m/kf_attention_2/value_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/v/kf_attention_2/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/kf_attention_2/query_kernel
Ю
6Adam/v/kf_attention_2/query_kernel/Read/ReadVariableOpReadVariableOp"Adam/v/kf_attention_2/query_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/m/kf_attention_2/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/kf_attention_2/query_kernel
Ю
6Adam/m/kf_attention_2/query_kernel/Read/ReadVariableOpReadVariableOp"Adam/m/kf_attention_2/query_kernel*#
_output_shapes
:А*
dtype0
Ъ
!Adam/v/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_4/beta
У
5Adam/v/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_4/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_4/beta
У
5Adam/m/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_4/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_4/gamma
Х
6Adam/v/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_4/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_4/gamma
Х
6Adam/m/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_4/gamma*
_output_shapes
:*
dtype0
А
Adam/v/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_3/bias
y
(Adam/v/conv1d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_3/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_3/bias
y
(Adam/m/conv1d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_3/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_3/kernel
Е
*Adam/v/conv1d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_3/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_3/kernel
Е
*Adam/m/conv1d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_3/kernel*"
_output_shapes
:*
dtype0
А
Adam/v/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_2/bias
y
(Adam/v/conv1d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_2/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_2/bias
y
(Adam/m/conv1d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_2/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_2/kernel
Е
*Adam/v/conv1d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_2/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_2/kernel
Е
*Adam/m/conv1d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_2/kernel*"
_output_shapes
:*
dtype0
Ъ
!Adam/v/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_3/beta
У
5Adam/v/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_3/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_3/beta
У
5Adam/m/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_3/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_3/gamma
Х
6Adam/v/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_3/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_3/gamma
Х
6Adam/m/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_3/gamma*
_output_shapes
:*
dtype0
в
%Adam/v/kf_attention_1/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/kf_attention_1/projection_bias
Ы
9Adam/v/kf_attention_1/projection_bias/Read/ReadVariableOpReadVariableOp%Adam/v/kf_attention_1/projection_bias*
_output_shapes
:*
dtype0
в
%Adam/m/kf_attention_1/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/kf_attention_1/projection_bias
Ы
9Adam/m/kf_attention_1/projection_bias/Read/ReadVariableOpReadVariableOp%Adam/m/kf_attention_1/projection_bias*
_output_shapes
:*
dtype0
п
'Adam/v/kf_attention_1/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adam/v/kf_attention_1/projection_kernel
и
;Adam/v/kf_attention_1/projection_kernel/Read/ReadVariableOpReadVariableOp'Adam/v/kf_attention_1/projection_kernel*#
_output_shapes
:А*
dtype0
п
'Adam/m/kf_attention_1/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adam/m/kf_attention_1/projection_kernel
и
;Adam/m/kf_attention_1/projection_kernel/Read/ReadVariableOpReadVariableOp'Adam/m/kf_attention_1/projection_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/v/kf_attention_1/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/kf_attention_1/value_kernel
Ю
6Adam/v/kf_attention_1/value_kernel/Read/ReadVariableOpReadVariableOp"Adam/v/kf_attention_1/value_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/m/kf_attention_1/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/kf_attention_1/value_kernel
Ю
6Adam/m/kf_attention_1/value_kernel/Read/ReadVariableOpReadVariableOp"Adam/m/kf_attention_1/value_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/v/kf_attention_1/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/kf_attention_1/query_kernel
Ю
6Adam/v/kf_attention_1/query_kernel/Read/ReadVariableOpReadVariableOp"Adam/v/kf_attention_1/query_kernel*#
_output_shapes
:А*
dtype0
е
"Adam/m/kf_attention_1/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/kf_attention_1/query_kernel
Ю
6Adam/m/kf_attention_1/query_kernel/Read/ReadVariableOpReadVariableOp"Adam/m/kf_attention_1/query_kernel*#
_output_shapes
:А*
dtype0
Ъ
!Adam/v/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_2/beta
У
5Adam/v/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_2/beta
У
5Adam/m/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_2/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_2/gamma
Х
6Adam/v/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_2/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_2/gamma
Х
6Adam/m/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_2/gamma*
_output_shapes
:*
dtype0
А
Adam/v/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d_1/bias
y
(Adam/v/conv1d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_1/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d_1/bias
y
(Adam/m/conv1d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_1/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv1d_1/kernel
Е
*Adam/v/conv1d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d_1/kernel*"
_output_shapes
:*
dtype0
М
Adam/m/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv1d_1/kernel
Е
*Adam/m/conv1d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d_1/kernel*"
_output_shapes
:*
dtype0
|
Adam/v/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv1d/bias
u
&Adam/v/conv1d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv1d/bias
u
&Adam/m/conv1d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d/bias*
_output_shapes
:*
dtype0
И
Adam/v/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv1d/kernel
Б
(Adam/v/conv1d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d/kernel*"
_output_shapes
:*
dtype0
И
Adam/m/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv1d/kernel
Б
(Adam/m/conv1d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d/kernel*"
_output_shapes
:*
dtype0
Ъ
!Adam/v/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/layer_normalization_1/beta
У
5Adam/v/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/layer_normalization_1/beta
У
5Adam/m/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_1/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_1/gamma
Х
6Adam/v/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_1/gamma
Х
6Adam/m/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
Ю
#Adam/v/kf_attention/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/kf_attention/projection_bias
Ч
7Adam/v/kf_attention/projection_bias/Read/ReadVariableOpReadVariableOp#Adam/v/kf_attention/projection_bias*
_output_shapes
:*
dtype0
Ю
#Adam/m/kf_attention/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/kf_attention/projection_bias
Ч
7Adam/m/kf_attention/projection_bias/Read/ReadVariableOpReadVariableOp#Adam/m/kf_attention/projection_bias*
_output_shapes
:*
dtype0
л
%Adam/v/kf_attention/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%Adam/v/kf_attention/projection_kernel
д
9Adam/v/kf_attention/projection_kernel/Read/ReadVariableOpReadVariableOp%Adam/v/kf_attention/projection_kernel*#
_output_shapes
:А*
dtype0
л
%Adam/m/kf_attention/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%Adam/m/kf_attention/projection_kernel
д
9Adam/m/kf_attention/projection_kernel/Read/ReadVariableOpReadVariableOp%Adam/m/kf_attention/projection_kernel*#
_output_shapes
:А*
dtype0
б
 Adam/v/kf_attention/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/v/kf_attention/value_kernel
Ъ
4Adam/v/kf_attention/value_kernel/Read/ReadVariableOpReadVariableOp Adam/v/kf_attention/value_kernel*#
_output_shapes
:А*
dtype0
б
 Adam/m/kf_attention/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/m/kf_attention/value_kernel
Ъ
4Adam/m/kf_attention/value_kernel/Read/ReadVariableOpReadVariableOp Adam/m/kf_attention/value_kernel*#
_output_shapes
:А*
dtype0
б
 Adam/v/kf_attention/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/v/kf_attention/query_kernel
Ъ
4Adam/v/kf_attention/query_kernel/Read/ReadVariableOpReadVariableOp Adam/v/kf_attention/query_kernel*#
_output_shapes
:А*
dtype0
б
 Adam/m/kf_attention/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/m/kf_attention/query_kernel
Ъ
4Adam/m/kf_attention/query_kernel/Read/ReadVariableOpReadVariableOp Adam/m/kf_attention/query_kernel*#
_output_shapes
:А*
dtype0
Ц
Adam/v/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/layer_normalization/beta
П
3Adam/v/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/layer_normalization/beta*
_output_shapes
:*
dtype0
Ц
Adam/m/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/layer_normalization/beta
П
3Adam/m/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/layer_normalization/beta*
_output_shapes
:*
dtype0
Ш
 Adam/v/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/layer_normalization/gamma
С
4Adam/v/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/layer_normalization/gamma*
_output_shapes
:*
dtype0
Ш
 Adam/m/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/layer_normalization/gamma
С
4Adam/m/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/layer_normalization/gamma*
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:┘*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:┘*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А┘*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
А┘*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
А*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	
А*
dtype0
r
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_7/bias
k
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
_output_shapes
:*
dtype0
~
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_7/kernel
w
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*"
_output_shapes
:*
dtype0
r
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_6/bias
k
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes
:*
dtype0
~
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_6/kernel
w
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*"
_output_shapes
:*
dtype0
М
layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_7/beta
Е
.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
:*
dtype0
О
layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_7/gamma
З
/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
:*
dtype0
Ф
kf_attention_3/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name kf_attention_3/projection_bias
Н
2kf_attention_3/projection_bias/Read/ReadVariableOpReadVariableOpkf_attention_3/projection_bias*
_output_shapes
:*
dtype0
б
 kf_attention_3/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" kf_attention_3/projection_kernel
Ъ
4kf_attention_3/projection_kernel/Read/ReadVariableOpReadVariableOp kf_attention_3/projection_kernel*#
_output_shapes
:А*
dtype0
Ч
kf_attention_3/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namekf_attention_3/value_kernel
Р
/kf_attention_3/value_kernel/Read/ReadVariableOpReadVariableOpkf_attention_3/value_kernel*#
_output_shapes
:А*
dtype0
Ч
kf_attention_3/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namekf_attention_3/query_kernel
Р
/kf_attention_3/query_kernel/Read/ReadVariableOpReadVariableOpkf_attention_3/query_kernel*#
_output_shapes
:А*
dtype0
М
layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_6/beta
Е
.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
:*
dtype0
О
layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_6/gamma
З
/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
:*
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:*
dtype0
~
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_5/kernel
w
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*"
_output_shapes
:*
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
:*
dtype0
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:*
dtype0
М
layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_5/beta
Е
.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
:*
dtype0
О
layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_5/gamma
З
/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
:*
dtype0
Ф
kf_attention_2/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name kf_attention_2/projection_bias
Н
2kf_attention_2/projection_bias/Read/ReadVariableOpReadVariableOpkf_attention_2/projection_bias*
_output_shapes
:*
dtype0
б
 kf_attention_2/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" kf_attention_2/projection_kernel
Ъ
4kf_attention_2/projection_kernel/Read/ReadVariableOpReadVariableOp kf_attention_2/projection_kernel*#
_output_shapes
:А*
dtype0
Ч
kf_attention_2/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namekf_attention_2/value_kernel
Р
/kf_attention_2/value_kernel/Read/ReadVariableOpReadVariableOpkf_attention_2/value_kernel*#
_output_shapes
:А*
dtype0
Ч
kf_attention_2/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namekf_attention_2/query_kernel
Р
/kf_attention_2/query_kernel/Read/ReadVariableOpReadVariableOpkf_attention_2/query_kernel*#
_output_shapes
:А*
dtype0
М
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_4/beta
Е
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:*
dtype0
О
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_4/gamma
З
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
М
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_3/beta
Е
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:*
dtype0
О
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_3/gamma
З
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:*
dtype0
Ф
kf_attention_1/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name kf_attention_1/projection_bias
Н
2kf_attention_1/projection_bias/Read/ReadVariableOpReadVariableOpkf_attention_1/projection_bias*
_output_shapes
:*
dtype0
б
 kf_attention_1/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" kf_attention_1/projection_kernel
Ъ
4kf_attention_1/projection_kernel/Read/ReadVariableOpReadVariableOp kf_attention_1/projection_kernel*#
_output_shapes
:А*
dtype0
Ч
kf_attention_1/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namekf_attention_1/value_kernel
Р
/kf_attention_1/value_kernel/Read/ReadVariableOpReadVariableOpkf_attention_1/value_kernel*#
_output_shapes
:А*
dtype0
Ч
kf_attention_1/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namekf_attention_1/query_kernel
Р
/kf_attention_1/query_kernel/Read/ReadVariableOpReadVariableOpkf_attention_1/query_kernel*#
_output_shapes
:А*
dtype0
М
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_2/beta
Е
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:*
dtype0
О
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_2/gamma
З
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
М
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_1/beta
Е
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:*
dtype0
О
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_1/gamma
З
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:*
dtype0
Р
kf_attention/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namekf_attention/projection_bias
Й
0kf_attention/projection_bias/Read/ReadVariableOpReadVariableOpkf_attention/projection_bias*
_output_shapes
:*
dtype0
Э
kf_attention/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name kf_attention/projection_kernel
Ц
2kf_attention/projection_kernel/Read/ReadVariableOpReadVariableOpkf_attention/projection_kernel*#
_output_shapes
:А*
dtype0
У
kf_attention/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namekf_attention/value_kernel
М
-kf_attention/value_kernel/Read/ReadVariableOpReadVariableOpkf_attention/value_kernel*#
_output_shapes
:А*
dtype0
У
kf_attention/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namekf_attention/query_kernel
М
-kf_attention/query_kernel/Read/ReadVariableOpReadVariableOpkf_attention/query_kernel*#
_output_shapes
:А*
dtype0
И
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelayer_normalization/beta
Б
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:*
dtype0
К
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelayer_normalization/gamma
Г
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:*
dtype0
В
serving_default_input_1Placeholder*+
_output_shapes
:         
*
dtype0* 
shape:         

√
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer_normalization/gammalayer_normalization/betakf_attention/query_kernelkf_attention/value_kernelkf_attention/projection_kernelkf_attention/projection_biaslayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslayer_normalization_2/gammalayer_normalization_2/betakf_attention_1/query_kernelkf_attention_1/value_kernel kf_attention_1/projection_kernelkf_attention_1/projection_biaslayer_normalization_3/gammalayer_normalization_3/betaconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biaslayer_normalization_4/gammalayer_normalization_4/betakf_attention_2/query_kernelkf_attention_2/value_kernel kf_attention_2/projection_kernelkf_attention_2/projection_biaslayer_normalization_5/gammalayer_normalization_5/betaconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biaslayer_normalization_6/gammalayer_normalization_6/betakf_attention_3/query_kernelkf_attention_3/value_kernel kf_attention_3/projection_kernelkf_attention_3/projection_biaslayer_normalization_7/gammalayer_normalization_7/betaconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_532285

NoOpNoOp
▀╩
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Щ╩
valueО╩BК╩ BВ╩
╚

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
layer-25
layer_with_weights-14
layer-26
layer-27
layer_with_weights-15
layer-28
layer_with_weights-16
layer-29
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"layer_with_weights-18
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer-39
)layer_with_weights-21
)layer-40
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_default_save_signature
1	optimizer
2
signatures*
* 
п
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9axis
	:gamma
;beta*
э
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
Bdropout
Cquery_kernel
Dvalue_kernel
Eprojection_kernel
Fprojection_bias*
е
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator* 

N	keras_api* 
п
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta*
╚
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias
 `_jit_compiled_convolution_op*
е
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator* 
╚
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias
 p_jit_compiled_convolution_op*

q	keras_api* 
п
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xaxis
	ygamma
zbeta*
є
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses
Бdropout
Вquery_kernel
Гvalue_kernel
Дprojection_kernel
Еprojection_bias*
м
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses
М_random_generator* 

Н	keras_api* 
╕
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses
	Фaxis

Хgamma
	Цbeta*
╤
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
Эkernel
	Юbias
!Я_jit_compiled_convolution_op*
м
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
ж_random_generator* 
╤
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
нkernel
	оbias
!п_jit_compiled_convolution_op*

░	keras_api* 
╕
▒	variables
▓trainable_variables
│regularization_losses
┤	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses
	╖axis

╕gamma
	╣beta*
°
║	variables
╗trainable_variables
╝regularization_losses
╜	keras_api
╛__call__
+┐&call_and_return_all_conditional_losses
└dropout
┴query_kernel
┬value_kernel
├projection_kernel
─projection_bias*
м
┼	variables
╞trainable_variables
╟regularization_losses
╚	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses
╦_random_generator* 

╠	keras_api* 
╕
═	variables
╬trainable_variables
╧regularization_losses
╨	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses
	╙axis

╘gamma
	╒beta*
╤
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses
▄kernel
	▌bias
!▐_jit_compiled_convolution_op*
м
▀	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
х_random_generator* 
╤
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses
ьkernel
	эbias
!ю_jit_compiled_convolution_op*

я	keras_api* 
╕
Ё	variables
ёtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses
	Ўaxis

ўgamma
	°beta*
°
∙	variables
·trainable_variables
√regularization_losses
№	keras_api
¤__call__
+■&call_and_return_all_conditional_losses
 dropout
Аquery_kernel
Бvalue_kernel
Вprojection_kernel
Гprojection_bias*
м
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
К_random_generator* 

Л	keras_api* 
╕
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
	Тaxis

Уgamma
	Фbeta*
╤
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias
!Э_jit_compiled_convolution_op*
м
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses
д_random_generator* 
╤
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
лkernel
	мbias
!н_jit_compiled_convolution_op*

о	keras_api* 
Ф
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
│__call__
+┤&call_and_return_all_conditional_losses* 
о
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗kernel
	╝bias*
м
╜	variables
╛trainable_variables
┐regularization_losses
└	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses
├_random_generator* 
о
─	variables
┼trainable_variables
╞regularization_losses
╟	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses
╩kernel
	╦bias*
└
:0
;1
C2
D3
E4
F5
V6
W7
^8
_9
n10
o11
y12
z13
В14
Г15
Д16
Е17
Х18
Ц19
Э20
Ю21
н22
о23
╕24
╣25
┴26
┬27
├28
─29
╘30
╒31
▄32
▌33
ь34
э35
ў36
°37
А38
Б39
В40
Г41
У42
Ф43
Ы44
Ь45
л46
м47
╗48
╝49
╩50
╦51*
└
:0
;1
C2
D3
E4
F5
V6
W7
^8
_9
n10
o11
y12
z13
В14
Г15
Д16
Е17
Х18
Ц19
Э20
Ю21
н22
о23
╕24
╣25
┴26
┬27
├28
─29
╘30
╒31
▄32
▌33
ь34
э35
ў36
°37
А38
Б39
В40
Г41
У42
Ф43
Ы44
Ь45
л46
м47
╗48
╝49
╩50
╦51*
* 
╡
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
:
╤trace_0
╥trace_1
╙trace_2
╘trace_3* 
:
╒trace_0
╓trace_1
╫trace_2
╪trace_3* 
* 
И
┘
_variables
┌_iterations
█_learning_rate
▄_index_dict
▌
_momentums
▐_velocities
▀_update_step_xla*

рserving_default* 

:0
;1*

:0
;1*
* 
Ш
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

цtrace_0* 

чtrace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
 
C0
D1
E2
F3*
 
C0
D1
E2
F3*
* 
Ш
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

эtrace_0
юtrace_1* 

яtrace_0
Ёtrace_1* 
*
ё	keras_api
Є_random_generator* 
oi
VARIABLE_VALUEkf_attention/query_kernel<layer_with_weights-1/query_kernel/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEkf_attention/value_kernel<layer_with_weights-1/value_kernel/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEkf_attention/projection_kernelAlayer_with_weights-1/projection_kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEkf_attention/projection_bias?layer_with_weights-1/projection_bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
єnon_trainable_variables
Їlayers
їmetrics
 Ўlayer_regularization_losses
ўlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

°trace_0
∙trace_1* 

·trace_0
√trace_1* 
* 
* 

V0
W1*

V0
W1*
* 
Ш
№non_trainable_variables
¤layers
■metrics
  layer_regularization_losses
Аlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

Бtrace_0* 

Вtrace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*

^0
_1*

^0
_1*
* 
Ш
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

Иtrace_0* 

Йtrace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

Пtrace_0
Рtrace_1* 

Сtrace_0
Тtrace_1* 
* 

n0
o1*

n0
o1*
* 
Ш
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

Шtrace_0* 

Щtrace_0* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

y0
z1*

y0
z1*
* 
Ш
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

Яtrace_0* 

аtrace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
$
В0
Г1
Д2
Е3*
$
В0
Г1
Д2
Е3*
* 
Ъ
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

жtrace_0
зtrace_1* 

иtrace_0
йtrace_1* 
*
к	keras_api
л_random_generator* 
qk
VARIABLE_VALUEkf_attention_1/query_kernel<layer_with_weights-6/query_kernel/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEkf_attention_1/value_kernel<layer_with_weights-6/value_kernel/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE kf_attention_1/projection_kernelAlayer_with_weights-6/projection_kernel/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEkf_attention_1/projection_bias?layer_with_weights-6/projection_bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 

▒trace_0
▓trace_1* 

│trace_0
┤trace_1* 
* 
* 

Х0
Ц1*

Х0
Ц1*
* 
Ю
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*

║trace_0* 

╗trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*

Э0
Ю1*

Э0
Ю1*
* 
Ю
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*

┴trace_0* 

┬trace_0* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses* 

╚trace_0
╔trace_1* 

╩trace_0
╦trace_1* 
* 

н0
о1*

н0
о1*
* 
Ю
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses*

╤trace_0* 

╥trace_0* 
_Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

╕0
╣1*

╕0
╣1*
* 
Ю
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
▒	variables
▓trainable_variables
│regularization_losses
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses*

╪trace_0* 

┘trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
$
┴0
┬1
├2
─3*
$
┴0
┬1
├2
─3*
* 
Ю
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
║	variables
╗trainable_variables
╝regularization_losses
╛__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses*

▀trace_0
рtrace_1* 

сtrace_0
тtrace_1* 
*
у	keras_api
ф_random_generator* 
rl
VARIABLE_VALUEkf_attention_2/query_kernel=layer_with_weights-11/query_kernel/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEkf_attention_2/value_kernel=layer_with_weights-11/value_kernel/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE kf_attention_2/projection_kernelBlayer_with_weights-11/projection_kernel/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEkf_attention_2/projection_bias@layer_with_weights-11/projection_bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
┼	variables
╞trainable_variables
╟regularization_losses
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses* 

ъtrace_0
ыtrace_1* 

ьtrace_0
эtrace_1* 
* 
* 

╘0
╒1*

╘0
╒1*
* 
Ю
юnon_trainable_variables
яlayers
Ёmetrics
 ёlayer_regularization_losses
Єlayer_metrics
═	variables
╬trainable_variables
╧regularization_losses
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses*

єtrace_0* 

Їtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_5/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_5/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*

▄0
▌1*

▄0
▌1*
* 
Ю
їnon_trainable_variables
Ўlayers
ўmetrics
 °layer_regularization_losses
∙layer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses*

·trace_0* 

√trace_0* 
`Z
VARIABLE_VALUEconv1d_4/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_4/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
№non_trainable_variables
¤layers
■metrics
  layer_regularization_losses
Аlayer_metrics
▀	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses* 

Бtrace_0
Вtrace_1* 

Гtrace_0
Дtrace_1* 
* 

ь0
э1*

ь0
э1*
* 
Ю
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
`Z
VARIABLE_VALUEconv1d_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

ў0
°1*

ў0
°1*
* 
Ю
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
Ё	variables
ёtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses*

Сtrace_0* 

Тtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_6/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_6/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
$
А0
Б1
В2
Г3*
$
А0
Б1
В2
Г3*
* 
Ю
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
∙	variables
·trainable_variables
√regularization_losses
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses*

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
*
Ь	keras_api
Э_random_generator* 
rl
VARIABLE_VALUEkf_attention_3/query_kernel=layer_with_weights-16/query_kernel/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEkf_attention_3/value_kernel=layer_with_weights-16/value_kernel/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE kf_attention_3/projection_kernelBlayer_with_weights-16/projection_kernel/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEkf_attention_3/projection_bias@layer_with_weights-16/projection_bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses* 

гtrace_0
дtrace_1* 

еtrace_0
жtrace_1* 
* 
* 

У0
Ф1*

У0
Ф1*
* 
Ю
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

мtrace_0* 

нtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_7/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_7/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*

Ы0
Ь1*

Ы0
Ь1*
* 
Ю
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses*

│trace_0* 

┤trace_0* 
`Z
VARIABLE_VALUEconv1d_6/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_6/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses* 

║trace_0
╗trace_1* 

╝trace_0
╜trace_1* 
* 

л0
м1*

л0
м1*
* 
Ю
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses*

├trace_0* 

─trace_0* 
`Z
VARIABLE_VALUEconv1d_7/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_7/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
Ь
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
п	variables
░trainable_variables
▒regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses* 

╩trace_0* 

╦trace_0* 

╗0
╝1*

╗0
╝1*
* 
Ю
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses*

╤trace_0* 

╥trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
╜	variables
╛trainable_variables
┐regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses* 

╪trace_0
┘trace_1* 

┌trace_0
█trace_1* 
* 

╩0
╦1*

╩0
╦1*
* 
Ю
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
─	variables
┼trainable_variables
╞regularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses*

сtrace_0* 

тtrace_0* 
_Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
┬
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40*

у0
ф1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
░
┌0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
°20
∙21
·22
√23
№24
¤25
■26
 27
А28
Б29
В30
Г31
Д32
Е33
Ж34
З35
И36
Й37
К38
Л39
М40
Н41
О42
П43
Р44
С45
Т46
У47
Ф48
Х49
Ц50
Ч51
Ш52
Щ53
Ъ54
Ы55
Ь56
Э57
Ю58
Я59
а60
б61
в62
г63
д64
е65
ж66
з67
и68
й69
к70
л71
м72
н73
о74
п75
░76
▒77
▓78
│79
┤80
╡81
╢82
╖83
╕84
╣85
║86
╗87
╝88
╜89
╛90
┐91
└92
┴93
┬94
├95
─96
┼97
╞98
╟99
╚100
╔101
╩102
╦103
╠104*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
╬
х0
ч1
щ2
ы3
э4
я5
ё6
є7
ї8
ў9
∙10
√11
¤12
 13
Б14
Г15
Е16
З17
Й18
Л19
Н20
П21
С22
У23
Х24
Ч25
Щ26
Ы27
Э28
Я29
б30
г31
е32
з33
й34
л35
н36
п37
▒38
│39
╡40
╖41
╣42
╗43
╜44
┐45
┴46
├47
┼48
╟49
╔50
╦51*
╬
ц0
ш1
ъ2
ь3
ю4
Ё5
Є6
Ї7
Ў8
°9
·10
№11
■12
А13
В14
Д15
Ж16
И17
К18
М19
О20
Р21
Т22
Ф23
Ц24
Ш25
Ъ26
Ь27
Ю28
а29
в30
д31
ж32
и33
к34
м35
о36
░37
▓38
┤39
╢40
╕41
║42
╝43
╛44
└45
┬46
─47
╞48
╚49
╩50
╠51*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
B0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Б0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


└0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


 0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
═	variables
╬	keras_api

╧total

╨count*
M
╤	variables
╥	keras_api

╙total

╘count
╒
_fn_kwargs*
ke
VARIABLE_VALUE Adam/m/layer_normalization/gamma1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/layer_normalization/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/layer_normalization/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/layer_normalization/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/kf_attention/query_kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/kf_attention/query_kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/kf_attention/value_kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/kf_attention/value_kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/kf_attention/projection_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/kf_attention/projection_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/kf_attention/projection_bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/kf_attention/projection_bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv1d/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv1d/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_1/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_1/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_1/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_1/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_2/gamma2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_2/gamma2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_2/beta2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_2/beta2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/kf_attention_1/query_kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/kf_attention_1/query_kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/kf_attention_1/value_kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/kf_attention_1/value_kernel2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/kf_attention_1/projection_kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/kf_attention_1/projection_kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/kf_attention_1/projection_bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/kf_attention_1/projection_bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_3/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_3/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_3/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_3/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_2/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_2/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_2/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_2/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_3/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_3/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_3/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_3/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_4/gamma2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_4/gamma2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_4/beta2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_4/beta2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/kf_attention_2/query_kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/kf_attention_2/query_kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/kf_attention_2/value_kernel2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/kf_attention_2/value_kernel2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/kf_attention_2/projection_kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/kf_attention_2/projection_kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/kf_attention_2/projection_bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/kf_attention_2/projection_bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_5/gamma2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_5/gamma2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_5/beta2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_5/beta2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_4/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_4/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_4/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_4/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_5/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_5/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_5/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_5/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_6/gamma2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_6/gamma2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_6/beta2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_6/beta2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/kf_attention_3/query_kernel2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/kf_attention_3/query_kernel2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/kf_attention_3/value_kernel2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/kf_attention_3/value_kernel2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/kf_attention_3/projection_kernel2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/kf_attention_3/projection_kernel2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/kf_attention_3/projection_bias2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/kf_attention_3/projection_bias2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_7/gamma2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_7/gamma2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_7/beta2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_7/beta2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_6/kernel2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_6/kernel2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_6/bias2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_6/bias2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv1d_7/kernel2optimizer/_variables/93/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv1d_7/kernel2optimizer/_variables/94/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv1d_7/bias2optimizer/_variables/95/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv1d_7/bias2optimizer/_variables/96/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/97/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/98/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/99/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/bias3optimizer/_variables/100/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_1/kernel3optimizer/_variables/101/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_1/kernel3optimizer/_variables/102/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/bias3optimizer/_variables/103/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/bias3optimizer/_variables/104/.ATTRIBUTES/VARIABLE_VALUE*

╧0
╨1*

═	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

╙0
╘1*

╤	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы@
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp-kf_attention/query_kernel/Read/ReadVariableOp-kf_attention/value_kernel/Read/ReadVariableOp2kf_attention/projection_kernel/Read/ReadVariableOp0kf_attention/projection_bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp/kf_attention_1/query_kernel/Read/ReadVariableOp/kf_attention_1/value_kernel/Read/ReadVariableOp4kf_attention_1/projection_kernel/Read/ReadVariableOp2kf_attention_1/projection_bias/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp/kf_attention_2/query_kernel/Read/ReadVariableOp/kf_attention_2/value_kernel/Read/ReadVariableOp4kf_attention_2/projection_kernel/Read/ReadVariableOp2kf_attention_2/projection_bias/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp/layer_normalization_6/gamma/Read/ReadVariableOp.layer_normalization_6/beta/Read/ReadVariableOp/kf_attention_3/query_kernel/Read/ReadVariableOp/kf_attention_3/value_kernel/Read/ReadVariableOp4kf_attention_3/projection_kernel/Read/ReadVariableOp2kf_attention_3/projection_bias/Read/ReadVariableOp/layer_normalization_7/gamma/Read/ReadVariableOp.layer_normalization_7/beta/Read/ReadVariableOp#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp4Adam/m/layer_normalization/gamma/Read/ReadVariableOp4Adam/v/layer_normalization/gamma/Read/ReadVariableOp3Adam/m/layer_normalization/beta/Read/ReadVariableOp3Adam/v/layer_normalization/beta/Read/ReadVariableOp4Adam/m/kf_attention/query_kernel/Read/ReadVariableOp4Adam/v/kf_attention/query_kernel/Read/ReadVariableOp4Adam/m/kf_attention/value_kernel/Read/ReadVariableOp4Adam/v/kf_attention/value_kernel/Read/ReadVariableOp9Adam/m/kf_attention/projection_kernel/Read/ReadVariableOp9Adam/v/kf_attention/projection_kernel/Read/ReadVariableOp7Adam/m/kf_attention/projection_bias/Read/ReadVariableOp7Adam/v/kf_attention/projection_bias/Read/ReadVariableOp6Adam/m/layer_normalization_1/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_1/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_1/beta/Read/ReadVariableOp5Adam/v/layer_normalization_1/beta/Read/ReadVariableOp(Adam/m/conv1d/kernel/Read/ReadVariableOp(Adam/v/conv1d/kernel/Read/ReadVariableOp&Adam/m/conv1d/bias/Read/ReadVariableOp&Adam/v/conv1d/bias/Read/ReadVariableOp*Adam/m/conv1d_1/kernel/Read/ReadVariableOp*Adam/v/conv1d_1/kernel/Read/ReadVariableOp(Adam/m/conv1d_1/bias/Read/ReadVariableOp(Adam/v/conv1d_1/bias/Read/ReadVariableOp6Adam/m/layer_normalization_2/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_2/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_2/beta/Read/ReadVariableOp5Adam/v/layer_normalization_2/beta/Read/ReadVariableOp6Adam/m/kf_attention_1/query_kernel/Read/ReadVariableOp6Adam/v/kf_attention_1/query_kernel/Read/ReadVariableOp6Adam/m/kf_attention_1/value_kernel/Read/ReadVariableOp6Adam/v/kf_attention_1/value_kernel/Read/ReadVariableOp;Adam/m/kf_attention_1/projection_kernel/Read/ReadVariableOp;Adam/v/kf_attention_1/projection_kernel/Read/ReadVariableOp9Adam/m/kf_attention_1/projection_bias/Read/ReadVariableOp9Adam/v/kf_attention_1/projection_bias/Read/ReadVariableOp6Adam/m/layer_normalization_3/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_3/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_3/beta/Read/ReadVariableOp5Adam/v/layer_normalization_3/beta/Read/ReadVariableOp*Adam/m/conv1d_2/kernel/Read/ReadVariableOp*Adam/v/conv1d_2/kernel/Read/ReadVariableOp(Adam/m/conv1d_2/bias/Read/ReadVariableOp(Adam/v/conv1d_2/bias/Read/ReadVariableOp*Adam/m/conv1d_3/kernel/Read/ReadVariableOp*Adam/v/conv1d_3/kernel/Read/ReadVariableOp(Adam/m/conv1d_3/bias/Read/ReadVariableOp(Adam/v/conv1d_3/bias/Read/ReadVariableOp6Adam/m/layer_normalization_4/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_4/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_4/beta/Read/ReadVariableOp5Adam/v/layer_normalization_4/beta/Read/ReadVariableOp6Adam/m/kf_attention_2/query_kernel/Read/ReadVariableOp6Adam/v/kf_attention_2/query_kernel/Read/ReadVariableOp6Adam/m/kf_attention_2/value_kernel/Read/ReadVariableOp6Adam/v/kf_attention_2/value_kernel/Read/ReadVariableOp;Adam/m/kf_attention_2/projection_kernel/Read/ReadVariableOp;Adam/v/kf_attention_2/projection_kernel/Read/ReadVariableOp9Adam/m/kf_attention_2/projection_bias/Read/ReadVariableOp9Adam/v/kf_attention_2/projection_bias/Read/ReadVariableOp6Adam/m/layer_normalization_5/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_5/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_5/beta/Read/ReadVariableOp5Adam/v/layer_normalization_5/beta/Read/ReadVariableOp*Adam/m/conv1d_4/kernel/Read/ReadVariableOp*Adam/v/conv1d_4/kernel/Read/ReadVariableOp(Adam/m/conv1d_4/bias/Read/ReadVariableOp(Adam/v/conv1d_4/bias/Read/ReadVariableOp*Adam/m/conv1d_5/kernel/Read/ReadVariableOp*Adam/v/conv1d_5/kernel/Read/ReadVariableOp(Adam/m/conv1d_5/bias/Read/ReadVariableOp(Adam/v/conv1d_5/bias/Read/ReadVariableOp6Adam/m/layer_normalization_6/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_6/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_6/beta/Read/ReadVariableOp5Adam/v/layer_normalization_6/beta/Read/ReadVariableOp6Adam/m/kf_attention_3/query_kernel/Read/ReadVariableOp6Adam/v/kf_attention_3/query_kernel/Read/ReadVariableOp6Adam/m/kf_attention_3/value_kernel/Read/ReadVariableOp6Adam/v/kf_attention_3/value_kernel/Read/ReadVariableOp;Adam/m/kf_attention_3/projection_kernel/Read/ReadVariableOp;Adam/v/kf_attention_3/projection_kernel/Read/ReadVariableOp9Adam/m/kf_attention_3/projection_bias/Read/ReadVariableOp9Adam/v/kf_attention_3/projection_bias/Read/ReadVariableOp6Adam/m/layer_normalization_7/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_7/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_7/beta/Read/ReadVariableOp5Adam/v/layer_normalization_7/beta/Read/ReadVariableOp*Adam/m/conv1d_6/kernel/Read/ReadVariableOp*Adam/v/conv1d_6/kernel/Read/ReadVariableOp(Adam/m/conv1d_6/bias/Read/ReadVariableOp(Adam/v/conv1d_6/bias/Read/ReadVariableOp*Adam/m/conv1d_7/kernel/Read/ReadVariableOp*Adam/v/conv1d_7/kernel/Read/ReadVariableOp(Adam/m/conv1d_7/bias/Read/ReadVariableOp(Adam/v/conv1d_7/bias/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp)Adam/m/dense_1/kernel/Read/ReadVariableOp)Adam/v/dense_1/kernel/Read/ReadVariableOp'Adam/m/dense_1/bias/Read/ReadVariableOp'Adam/v/dense_1/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*▓
Tinк
з2д	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_534783
ю&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betakf_attention/query_kernelkf_attention/value_kernelkf_attention/projection_kernelkf_attention/projection_biaslayer_normalization_1/gammalayer_normalization_1/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslayer_normalization_2/gammalayer_normalization_2/betakf_attention_1/query_kernelkf_attention_1/value_kernel kf_attention_1/projection_kernelkf_attention_1/projection_biaslayer_normalization_3/gammalayer_normalization_3/betaconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biaslayer_normalization_4/gammalayer_normalization_4/betakf_attention_2/query_kernelkf_attention_2/value_kernel kf_attention_2/projection_kernelkf_attention_2/projection_biaslayer_normalization_5/gammalayer_normalization_5/betaconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biaslayer_normalization_6/gammalayer_normalization_6/betakf_attention_3/query_kernelkf_attention_3/value_kernel kf_attention_3/projection_kernelkf_attention_3/projection_biaslayer_normalization_7/gammalayer_normalization_7/betaconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationlearning_rate Adam/m/layer_normalization/gamma Adam/v/layer_normalization/gammaAdam/m/layer_normalization/betaAdam/v/layer_normalization/beta Adam/m/kf_attention/query_kernel Adam/v/kf_attention/query_kernel Adam/m/kf_attention/value_kernel Adam/v/kf_attention/value_kernel%Adam/m/kf_attention/projection_kernel%Adam/v/kf_attention/projection_kernel#Adam/m/kf_attention/projection_bias#Adam/v/kf_attention/projection_bias"Adam/m/layer_normalization_1/gamma"Adam/v/layer_normalization_1/gamma!Adam/m/layer_normalization_1/beta!Adam/v/layer_normalization_1/betaAdam/m/conv1d/kernelAdam/v/conv1d/kernelAdam/m/conv1d/biasAdam/v/conv1d/biasAdam/m/conv1d_1/kernelAdam/v/conv1d_1/kernelAdam/m/conv1d_1/biasAdam/v/conv1d_1/bias"Adam/m/layer_normalization_2/gamma"Adam/v/layer_normalization_2/gamma!Adam/m/layer_normalization_2/beta!Adam/v/layer_normalization_2/beta"Adam/m/kf_attention_1/query_kernel"Adam/v/kf_attention_1/query_kernel"Adam/m/kf_attention_1/value_kernel"Adam/v/kf_attention_1/value_kernel'Adam/m/kf_attention_1/projection_kernel'Adam/v/kf_attention_1/projection_kernel%Adam/m/kf_attention_1/projection_bias%Adam/v/kf_attention_1/projection_bias"Adam/m/layer_normalization_3/gamma"Adam/v/layer_normalization_3/gamma!Adam/m/layer_normalization_3/beta!Adam/v/layer_normalization_3/betaAdam/m/conv1d_2/kernelAdam/v/conv1d_2/kernelAdam/m/conv1d_2/biasAdam/v/conv1d_2/biasAdam/m/conv1d_3/kernelAdam/v/conv1d_3/kernelAdam/m/conv1d_3/biasAdam/v/conv1d_3/bias"Adam/m/layer_normalization_4/gamma"Adam/v/layer_normalization_4/gamma!Adam/m/layer_normalization_4/beta!Adam/v/layer_normalization_4/beta"Adam/m/kf_attention_2/query_kernel"Adam/v/kf_attention_2/query_kernel"Adam/m/kf_attention_2/value_kernel"Adam/v/kf_attention_2/value_kernel'Adam/m/kf_attention_2/projection_kernel'Adam/v/kf_attention_2/projection_kernel%Adam/m/kf_attention_2/projection_bias%Adam/v/kf_attention_2/projection_bias"Adam/m/layer_normalization_5/gamma"Adam/v/layer_normalization_5/gamma!Adam/m/layer_normalization_5/beta!Adam/v/layer_normalization_5/betaAdam/m/conv1d_4/kernelAdam/v/conv1d_4/kernelAdam/m/conv1d_4/biasAdam/v/conv1d_4/biasAdam/m/conv1d_5/kernelAdam/v/conv1d_5/kernelAdam/m/conv1d_5/biasAdam/v/conv1d_5/bias"Adam/m/layer_normalization_6/gamma"Adam/v/layer_normalization_6/gamma!Adam/m/layer_normalization_6/beta!Adam/v/layer_normalization_6/beta"Adam/m/kf_attention_3/query_kernel"Adam/v/kf_attention_3/query_kernel"Adam/m/kf_attention_3/value_kernel"Adam/v/kf_attention_3/value_kernel'Adam/m/kf_attention_3/projection_kernel'Adam/v/kf_attention_3/projection_kernel%Adam/m/kf_attention_3/projection_bias%Adam/v/kf_attention_3/projection_bias"Adam/m/layer_normalization_7/gamma"Adam/v/layer_normalization_7/gamma!Adam/m/layer_normalization_7/beta!Adam/v/layer_normalization_7/betaAdam/m/conv1d_6/kernelAdam/v/conv1d_6/kernelAdam/m/conv1d_6/biasAdam/v/conv1d_6/biasAdam/m/conv1d_7/kernelAdam/v/conv1d_7/kernelAdam/m/conv1d_7/biasAdam/v/conv1d_7/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcount*▒
Tinй
ж2г*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_535279ў·)
Ё
Р
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_530297

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
▒
G
+__inference_dropout_11_layer_call_fn_534150

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_530444d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_530193

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_530025

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╓
Ъ
)__inference_conv1d_2_layer_call_fn_533659

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_530047s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_2_cond_true_5301101
kf_attention_2_cond_530137:А1
kf_attention_2_cond_530139:А1
kf_attention_2_cond_530141:А(
kf_attention_2_cond_530143:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_530137kf_attention_2_cond_530139kf_attention_2_cond_530141kf_attention_2_cond_530143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530136п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

Б
c
*__inference_dropout_7_layer_call_fn_533837

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_530847s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╥
Ш
'__inference_conv1d_layer_call_fn_533424

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_529854s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_530411

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
Я
6__inference_layer_normalization_7_layer_call_fn_534098

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_530411s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╓
Ъ
)__inference_conv1d_4_layer_call_fn_533894

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_530240s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_530104

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_533384

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╓
Ъ
)__inference_conv1d_5_layer_call_fn_533946

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_530268s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Б
c
*__inference_dropout_4_layer_call_fn_533602

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_530960s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_3_cond_true_5314031
kf_attention_3_cond_531407:А1
kf_attention_3_cond_531409:А1
kf_attention_3_cond_531411:А(
kf_attention_3_cond_531413:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_531407kf_attention_3_cond_531409kf_attention_3_cond_531411kf_attention_3_cond_531413*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530329п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╚
У
D__inference_conv1d_6_layer_call_and_return_conditional_losses_534145

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┼
х
/__inference_kf_attention_1_layer_call_fn_533548

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529943s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_2_cond_true_5313461
kf_attention_2_cond_531350:А1
kf_attention_2_cond_531352:А1
kf_attention_2_cond_531354:А(
kf_attention_2_cond_531356:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_531350kf_attention_2_cond_531352kf_attention_2_cond_531354kf_attention_2_cond_531356*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530136п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╞
С
B__inference_conv1d_layer_call_and_return_conditional_losses_529854

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_529807

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
И
 
 kf_attention_3_cond_false_5303041
kf_attention_3_cond_530364:А1
kf_attention_3_cond_530366:А1
kf_attention_3_cond_530368:А(
kf_attention_3_cond_530370:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_530364kf_attention_3_cond_530366kf_attention_3_cond_530368kf_attention_3_cond_530370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530363п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╞
С
B__inference_conv1d_layer_call_and_return_conditional_losses_533440

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_533885

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_533690

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_1_cond_true_5299171
kf_attention_1_cond_529944:А1
kf_attention_1_cond_529946:А1
kf_attention_1_cond_529948:А(
kf_attention_1_cond_529950:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_529944kf_attention_1_cond_529946kf_attention_1_cond_529948kf_attention_1_cond_529950*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529943п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

У

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_534254

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ш
Я
6__inference_layer_normalization_5_layer_call_fn_533863

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_530218s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
И
 
 kf_attention_2_cond_false_5320511
kf_attention_2_cond_532066:А1
kf_attention_2_cond_532068:А1
kf_attention_2_cond_532070:А(
kf_attention_2_cond_532072:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_532066kf_attention_2_cond_532068kf_attention_2_cond_532070kf_attention_2_cond_532072*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530170п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

Н
╚
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533827

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┼
х
/__inference_kf_attention_2_layer_call_fn_533783

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530136s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
п
F
*__inference_dropout_2_layer_call_fn_533445

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_529865d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Г
d
+__inference_dropout_10_layer_call_fn_534072

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_530734s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╚
У
D__inference_conv1d_6_layer_call_and_return_conditional_losses_530433

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_529832

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529943

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
л

ў
C__inference_dense_1_layer_call_and_return_conditional_losses_534274

inputs2
matmul_readvariableop_resource:
А┘.
biasadd_readvariableop_resource:	┘
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А┘*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:┘*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:         ┘a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ┘w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_533650

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
в
╘
&__inference_model_layer_call_fn_530618
input_1
unknown:
	unknown_0: 
	unknown_1:А 
	unknown_2:А 
	unknown_3:А
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:!

unknown_13:А!

unknown_14:А!

unknown_15:А

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:!

unknown_25:А!

unknown_26:А!

unknown_27:А

unknown_28:

unknown_29:

unknown_30: 

unknown_31:

unknown_32: 

unknown_33:

unknown_34:

unknown_35:

unknown_36:!

unknown_37:А!

unknown_38:А!

unknown_39:А

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:	
А

unknown_48:	А

unknown_49:
А┘

unknown_50:	┘
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_530511p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         

!
_user_specified_name	input_1
И
 
 kf_attention_1_cond_false_5317501
kf_attention_1_cond_531765:А1
kf_attention_1_cond_531767:А1
kf_attention_1_cond_531769:А(
kf_attention_1_cond_531771:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_531765kf_attention_1_cond_531767kf_attention_1_cond_531769kf_attention_1_cond_531771*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529977п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

▌
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_534242

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Л
╞
H__inference_kf_attention_layer_call_and_return_conditional_losses_529784

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
У
D__inference_conv1d_7_layer_call_and_return_conditional_losses_530461

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
п
F
*__inference_dropout_1_layer_call_fn_533362

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_529807d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┼
х
/__inference_kf_attention_1_layer_call_fn_533535

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529977s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
щ
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_530444

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
▒
G
+__inference_dropout_10_layer_call_fn_534067

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_530386d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Б
c
*__inference_dropout_5_layer_call_fn_533685

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_530917s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╙
ь
kf_attention_cond_true_531692/
kf_attention_cond_531696:А/
kf_attention_cond_531698:А/
kf_attention_cond_531700:А&
kf_attention_cond_531702:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_531696kf_attention_cond_531698kf_attention_cond_531700kf_attention_cond_531702*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529750й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╓
Ъ
)__inference_conv1d_7_layer_call_fn_534181

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_530461s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
л

ў
C__inference_dense_1_layer_call_and_return_conditional_losses_530504

inputs2
matmul_readvariableop_resource:
А┘.
biasadd_readvariableop_resource:	┘
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А┘*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:┘*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:         ┘a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ┘w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┼
х
/__inference_kf_attention_3_layer_call_fn_534018

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530329s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
Я
6__inference_layer_normalization_3_layer_call_fn_533628

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_530025s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533805

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┤е
╝K
__inference__traced_save_534783
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop8
4savev2_kf_attention_query_kernel_read_readvariableop8
4savev2_kf_attention_value_kernel_read_readvariableop=
9savev2_kf_attention_projection_kernel_read_readvariableop;
7savev2_kf_attention_projection_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop:
6savev2_kf_attention_1_query_kernel_read_readvariableop:
6savev2_kf_attention_1_value_kernel_read_readvariableop?
;savev2_kf_attention_1_projection_kernel_read_readvariableop=
9savev2_kf_attention_1_projection_bias_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop:
6savev2_kf_attention_2_query_kernel_read_readvariableop:
6savev2_kf_attention_2_value_kernel_read_readvariableop?
;savev2_kf_attention_2_projection_kernel_read_readvariableop=
9savev2_kf_attention_2_projection_bias_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop:
6savev2_layer_normalization_6_gamma_read_readvariableop9
5savev2_layer_normalization_6_beta_read_readvariableop:
6savev2_kf_attention_3_query_kernel_read_readvariableop:
6savev2_kf_attention_3_value_kernel_read_readvariableop?
;savev2_kf_attention_3_projection_kernel_read_readvariableop=
9savev2_kf_attention_3_projection_bias_read_readvariableop:
6savev2_layer_normalization_7_gamma_read_readvariableop9
5savev2_layer_normalization_7_beta_read_readvariableop.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop?
;savev2_adam_m_layer_normalization_gamma_read_readvariableop?
;savev2_adam_v_layer_normalization_gamma_read_readvariableop>
:savev2_adam_m_layer_normalization_beta_read_readvariableop>
:savev2_adam_v_layer_normalization_beta_read_readvariableop?
;savev2_adam_m_kf_attention_query_kernel_read_readvariableop?
;savev2_adam_v_kf_attention_query_kernel_read_readvariableop?
;savev2_adam_m_kf_attention_value_kernel_read_readvariableop?
;savev2_adam_v_kf_attention_value_kernel_read_readvariableopD
@savev2_adam_m_kf_attention_projection_kernel_read_readvariableopD
@savev2_adam_v_kf_attention_projection_kernel_read_readvariableopB
>savev2_adam_m_kf_attention_projection_bias_read_readvariableopB
>savev2_adam_v_kf_attention_projection_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_1_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_1_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_1_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_1_beta_read_readvariableop3
/savev2_adam_m_conv1d_kernel_read_readvariableop3
/savev2_adam_v_conv1d_kernel_read_readvariableop1
-savev2_adam_m_conv1d_bias_read_readvariableop1
-savev2_adam_v_conv1d_bias_read_readvariableop5
1savev2_adam_m_conv1d_1_kernel_read_readvariableop5
1savev2_adam_v_conv1d_1_kernel_read_readvariableop3
/savev2_adam_m_conv1d_1_bias_read_readvariableop3
/savev2_adam_v_conv1d_1_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_2_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_2_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_2_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_2_beta_read_readvariableopA
=savev2_adam_m_kf_attention_1_query_kernel_read_readvariableopA
=savev2_adam_v_kf_attention_1_query_kernel_read_readvariableopA
=savev2_adam_m_kf_attention_1_value_kernel_read_readvariableopA
=savev2_adam_v_kf_attention_1_value_kernel_read_readvariableopF
Bsavev2_adam_m_kf_attention_1_projection_kernel_read_readvariableopF
Bsavev2_adam_v_kf_attention_1_projection_kernel_read_readvariableopD
@savev2_adam_m_kf_attention_1_projection_bias_read_readvariableopD
@savev2_adam_v_kf_attention_1_projection_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_3_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_3_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_3_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_3_beta_read_readvariableop5
1savev2_adam_m_conv1d_2_kernel_read_readvariableop5
1savev2_adam_v_conv1d_2_kernel_read_readvariableop3
/savev2_adam_m_conv1d_2_bias_read_readvariableop3
/savev2_adam_v_conv1d_2_bias_read_readvariableop5
1savev2_adam_m_conv1d_3_kernel_read_readvariableop5
1savev2_adam_v_conv1d_3_kernel_read_readvariableop3
/savev2_adam_m_conv1d_3_bias_read_readvariableop3
/savev2_adam_v_conv1d_3_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_4_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_4_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_4_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_4_beta_read_readvariableopA
=savev2_adam_m_kf_attention_2_query_kernel_read_readvariableopA
=savev2_adam_v_kf_attention_2_query_kernel_read_readvariableopA
=savev2_adam_m_kf_attention_2_value_kernel_read_readvariableopA
=savev2_adam_v_kf_attention_2_value_kernel_read_readvariableopF
Bsavev2_adam_m_kf_attention_2_projection_kernel_read_readvariableopF
Bsavev2_adam_v_kf_attention_2_projection_kernel_read_readvariableopD
@savev2_adam_m_kf_attention_2_projection_bias_read_readvariableopD
@savev2_adam_v_kf_attention_2_projection_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_5_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_5_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_5_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_5_beta_read_readvariableop5
1savev2_adam_m_conv1d_4_kernel_read_readvariableop5
1savev2_adam_v_conv1d_4_kernel_read_readvariableop3
/savev2_adam_m_conv1d_4_bias_read_readvariableop3
/savev2_adam_v_conv1d_4_bias_read_readvariableop5
1savev2_adam_m_conv1d_5_kernel_read_readvariableop5
1savev2_adam_v_conv1d_5_kernel_read_readvariableop3
/savev2_adam_m_conv1d_5_bias_read_readvariableop3
/savev2_adam_v_conv1d_5_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_6_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_6_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_6_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_6_beta_read_readvariableopA
=savev2_adam_m_kf_attention_3_query_kernel_read_readvariableopA
=savev2_adam_v_kf_attention_3_query_kernel_read_readvariableopA
=savev2_adam_m_kf_attention_3_value_kernel_read_readvariableopA
=savev2_adam_v_kf_attention_3_value_kernel_read_readvariableopF
Bsavev2_adam_m_kf_attention_3_projection_kernel_read_readvariableopF
Bsavev2_adam_v_kf_attention_3_projection_kernel_read_readvariableopD
@savev2_adam_m_kf_attention_3_projection_bias_read_readvariableopD
@savev2_adam_v_kf_attention_3_projection_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_7_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_7_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_7_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_7_beta_read_readvariableop5
1savev2_adam_m_conv1d_6_kernel_read_readvariableop5
1savev2_adam_v_conv1d_6_kernel_read_readvariableop3
/savev2_adam_m_conv1d_6_bias_read_readvariableop3
/savev2_adam_v_conv1d_6_bias_read_readvariableop5
1savev2_adam_m_conv1d_7_kernel_read_readvariableop5
1savev2_adam_v_conv1d_7_kernel_read_readvariableop3
/savev2_adam_m_conv1d_7_bias_read_readvariableop3
/savev2_adam_v_conv1d_7_bias_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop4
0savev2_adam_m_dense_1_kernel_read_readvariableop4
0savev2_adam_v_dense_1_kernel_read_readvariableop2
.savev2_adam_m_dense_1_bias_read_readvariableop2
.savev2_adam_v_dense_1_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╖E
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:г*
dtype0*▀D
value╒DB╥DгB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-6/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-6/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-6/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-11/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-11/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-11/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-16/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-16/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-16/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/93/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/94/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/95/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/96/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/97/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/98/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/99/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/100/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/101/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/102/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/103/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/104/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╕
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:г*
dtype0*▄
value╥B╧гB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╞H
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop4savev2_kf_attention_query_kernel_read_readvariableop4savev2_kf_attention_value_kernel_read_readvariableop9savev2_kf_attention_projection_kernel_read_readvariableop7savev2_kf_attention_projection_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop6savev2_kf_attention_1_query_kernel_read_readvariableop6savev2_kf_attention_1_value_kernel_read_readvariableop;savev2_kf_attention_1_projection_kernel_read_readvariableop9savev2_kf_attention_1_projection_bias_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop6savev2_kf_attention_2_query_kernel_read_readvariableop6savev2_kf_attention_2_value_kernel_read_readvariableop;savev2_kf_attention_2_projection_kernel_read_readvariableop9savev2_kf_attention_2_projection_bias_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop6savev2_layer_normalization_6_gamma_read_readvariableop5savev2_layer_normalization_6_beta_read_readvariableop6savev2_kf_attention_3_query_kernel_read_readvariableop6savev2_kf_attention_3_value_kernel_read_readvariableop;savev2_kf_attention_3_projection_kernel_read_readvariableop9savev2_kf_attention_3_projection_bias_read_readvariableop6savev2_layer_normalization_7_gamma_read_readvariableop5savev2_layer_normalization_7_beta_read_readvariableop*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop;savev2_adam_m_layer_normalization_gamma_read_readvariableop;savev2_adam_v_layer_normalization_gamma_read_readvariableop:savev2_adam_m_layer_normalization_beta_read_readvariableop:savev2_adam_v_layer_normalization_beta_read_readvariableop;savev2_adam_m_kf_attention_query_kernel_read_readvariableop;savev2_adam_v_kf_attention_query_kernel_read_readvariableop;savev2_adam_m_kf_attention_value_kernel_read_readvariableop;savev2_adam_v_kf_attention_value_kernel_read_readvariableop@savev2_adam_m_kf_attention_projection_kernel_read_readvariableop@savev2_adam_v_kf_attention_projection_kernel_read_readvariableop>savev2_adam_m_kf_attention_projection_bias_read_readvariableop>savev2_adam_v_kf_attention_projection_bias_read_readvariableop=savev2_adam_m_layer_normalization_1_gamma_read_readvariableop=savev2_adam_v_layer_normalization_1_gamma_read_readvariableop<savev2_adam_m_layer_normalization_1_beta_read_readvariableop<savev2_adam_v_layer_normalization_1_beta_read_readvariableop/savev2_adam_m_conv1d_kernel_read_readvariableop/savev2_adam_v_conv1d_kernel_read_readvariableop-savev2_adam_m_conv1d_bias_read_readvariableop-savev2_adam_v_conv1d_bias_read_readvariableop1savev2_adam_m_conv1d_1_kernel_read_readvariableop1savev2_adam_v_conv1d_1_kernel_read_readvariableop/savev2_adam_m_conv1d_1_bias_read_readvariableop/savev2_adam_v_conv1d_1_bias_read_readvariableop=savev2_adam_m_layer_normalization_2_gamma_read_readvariableop=savev2_adam_v_layer_normalization_2_gamma_read_readvariableop<savev2_adam_m_layer_normalization_2_beta_read_readvariableop<savev2_adam_v_layer_normalization_2_beta_read_readvariableop=savev2_adam_m_kf_attention_1_query_kernel_read_readvariableop=savev2_adam_v_kf_attention_1_query_kernel_read_readvariableop=savev2_adam_m_kf_attention_1_value_kernel_read_readvariableop=savev2_adam_v_kf_attention_1_value_kernel_read_readvariableopBsavev2_adam_m_kf_attention_1_projection_kernel_read_readvariableopBsavev2_adam_v_kf_attention_1_projection_kernel_read_readvariableop@savev2_adam_m_kf_attention_1_projection_bias_read_readvariableop@savev2_adam_v_kf_attention_1_projection_bias_read_readvariableop=savev2_adam_m_layer_normalization_3_gamma_read_readvariableop=savev2_adam_v_layer_normalization_3_gamma_read_readvariableop<savev2_adam_m_layer_normalization_3_beta_read_readvariableop<savev2_adam_v_layer_normalization_3_beta_read_readvariableop1savev2_adam_m_conv1d_2_kernel_read_readvariableop1savev2_adam_v_conv1d_2_kernel_read_readvariableop/savev2_adam_m_conv1d_2_bias_read_readvariableop/savev2_adam_v_conv1d_2_bias_read_readvariableop1savev2_adam_m_conv1d_3_kernel_read_readvariableop1savev2_adam_v_conv1d_3_kernel_read_readvariableop/savev2_adam_m_conv1d_3_bias_read_readvariableop/savev2_adam_v_conv1d_3_bias_read_readvariableop=savev2_adam_m_layer_normalization_4_gamma_read_readvariableop=savev2_adam_v_layer_normalization_4_gamma_read_readvariableop<savev2_adam_m_layer_normalization_4_beta_read_readvariableop<savev2_adam_v_layer_normalization_4_beta_read_readvariableop=savev2_adam_m_kf_attention_2_query_kernel_read_readvariableop=savev2_adam_v_kf_attention_2_query_kernel_read_readvariableop=savev2_adam_m_kf_attention_2_value_kernel_read_readvariableop=savev2_adam_v_kf_attention_2_value_kernel_read_readvariableopBsavev2_adam_m_kf_attention_2_projection_kernel_read_readvariableopBsavev2_adam_v_kf_attention_2_projection_kernel_read_readvariableop@savev2_adam_m_kf_attention_2_projection_bias_read_readvariableop@savev2_adam_v_kf_attention_2_projection_bias_read_readvariableop=savev2_adam_m_layer_normalization_5_gamma_read_readvariableop=savev2_adam_v_layer_normalization_5_gamma_read_readvariableop<savev2_adam_m_layer_normalization_5_beta_read_readvariableop<savev2_adam_v_layer_normalization_5_beta_read_readvariableop1savev2_adam_m_conv1d_4_kernel_read_readvariableop1savev2_adam_v_conv1d_4_kernel_read_readvariableop/savev2_adam_m_conv1d_4_bias_read_readvariableop/savev2_adam_v_conv1d_4_bias_read_readvariableop1savev2_adam_m_conv1d_5_kernel_read_readvariableop1savev2_adam_v_conv1d_5_kernel_read_readvariableop/savev2_adam_m_conv1d_5_bias_read_readvariableop/savev2_adam_v_conv1d_5_bias_read_readvariableop=savev2_adam_m_layer_normalization_6_gamma_read_readvariableop=savev2_adam_v_layer_normalization_6_gamma_read_readvariableop<savev2_adam_m_layer_normalization_6_beta_read_readvariableop<savev2_adam_v_layer_normalization_6_beta_read_readvariableop=savev2_adam_m_kf_attention_3_query_kernel_read_readvariableop=savev2_adam_v_kf_attention_3_query_kernel_read_readvariableop=savev2_adam_m_kf_attention_3_value_kernel_read_readvariableop=savev2_adam_v_kf_attention_3_value_kernel_read_readvariableopBsavev2_adam_m_kf_attention_3_projection_kernel_read_readvariableopBsavev2_adam_v_kf_attention_3_projection_kernel_read_readvariableop@savev2_adam_m_kf_attention_3_projection_bias_read_readvariableop@savev2_adam_v_kf_attention_3_projection_bias_read_readvariableop=savev2_adam_m_layer_normalization_7_gamma_read_readvariableop=savev2_adam_v_layer_normalization_7_gamma_read_readvariableop<savev2_adam_m_layer_normalization_7_beta_read_readvariableop<savev2_adam_v_layer_normalization_7_beta_read_readvariableop1savev2_adam_m_conv1d_6_kernel_read_readvariableop1savev2_adam_v_conv1d_6_kernel_read_readvariableop/savev2_adam_m_conv1d_6_bias_read_readvariableop/savev2_adam_v_conv1d_6_bias_read_readvariableop1savev2_adam_m_conv1d_7_kernel_read_readvariableop1savev2_adam_v_conv1d_7_kernel_read_readvariableop/savev2_adam_m_conv1d_7_bias_read_readvariableop/savev2_adam_v_conv1d_7_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *┤
dtypesй
ж2г	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*°
_input_shapesц
у: :::А:А:А::::::::::А:А:А::::::::::А:А:А::::::::::А:А:А::::::::	
А:А:
А┘:┘: : :::::А:А:А:А:А:А:::::::::::::::::::А:А:А:А:А:А:::::::::::::::::::А:А:А:А:А:А:::::::::::::::::::А:А:А:А:А:А:::::::::::::::	
А:	
А:А:А:
А┘:
А┘:┘:┘: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::)%
#
_output_shapes
:А:)%
#
_output_shapes
:А:)%
#
_output_shapes
:А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::(	$
"
_output_shapes
:: 


_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::)%
#
_output_shapes
:А:)%
#
_output_shapes
:А:)%
#
_output_shapes
:А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::)%
#
_output_shapes
:А:)%
#
_output_shapes
:А:)%
#
_output_shapes
:А: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
::(!$
"
_output_shapes
:: "

_output_shapes
::(#$
"
_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::)'%
#
_output_shapes
:А:)(%
#
_output_shapes
:А:))%
#
_output_shapes
:А: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::(-$
"
_output_shapes
:: .

_output_shapes
::(/$
"
_output_shapes
:: 0

_output_shapes
::%1!

_output_shapes
:	
А:!2

_output_shapes	
:А:&3"
 
_output_shapes
:
А┘:!4

_output_shapes	
:┘:5

_output_shapes
: :6

_output_shapes
: : 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
::);%
#
_output_shapes
:А:)<%
#
_output_shapes
:А:)=%
#
_output_shapes
:А:)>%
#
_output_shapes
:А:)?%
#
_output_shapes
:А:)@%
#
_output_shapes
:А: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
:: F

_output_shapes
::(G$
"
_output_shapes
::(H$
"
_output_shapes
:: I

_output_shapes
:: J

_output_shapes
::(K$
"
_output_shapes
::(L$
"
_output_shapes
:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
:: R

_output_shapes
::)S%
#
_output_shapes
:А:)T%
#
_output_shapes
:А:)U%
#
_output_shapes
:А:)V%
#
_output_shapes
:А:)W%
#
_output_shapes
:А:)X%
#
_output_shapes
:А: Y

_output_shapes
:: Z

_output_shapes
:: [

_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
:: ^

_output_shapes
::(_$
"
_output_shapes
::(`$
"
_output_shapes
:: a

_output_shapes
:: b

_output_shapes
::(c$
"
_output_shapes
::(d$
"
_output_shapes
:: e

_output_shapes
:: f

_output_shapes
:: g

_output_shapes
:: h

_output_shapes
:: i

_output_shapes
:: j

_output_shapes
::)k%
#
_output_shapes
:А:)l%
#
_output_shapes
:А:)m%
#
_output_shapes
:А:)n%
#
_output_shapes
:А:)o%
#
_output_shapes
:А:)p%
#
_output_shapes
:А: q

_output_shapes
:: r

_output_shapes
:: s

_output_shapes
:: t

_output_shapes
:: u

_output_shapes
:: v

_output_shapes
::(w$
"
_output_shapes
::(x$
"
_output_shapes
:: y

_output_shapes
:: z

_output_shapes
::({$
"
_output_shapes
::(|$
"
_output_shapes
:: }

_output_shapes
:: ~

_output_shapes
:: 

_output_shapes
::!А

_output_shapes
::!Б

_output_shapes
::!В

_output_shapes
::*Г%
#
_output_shapes
:А:*Д%
#
_output_shapes
:А:*Е%
#
_output_shapes
:А:*Ж%
#
_output_shapes
:А:*З%
#
_output_shapes
:А:*И%
#
_output_shapes
:А:!Й

_output_shapes
::!К

_output_shapes
::!Л

_output_shapes
::!М

_output_shapes
::!Н

_output_shapes
::!О

_output_shapes
::)П$
"
_output_shapes
::)Р$
"
_output_shapes
::!С

_output_shapes
::!Т

_output_shapes
::)У$
"
_output_shapes
::)Ф$
"
_output_shapes
::!Х

_output_shapes
::!Ц

_output_shapes
::&Ч!

_output_shapes
:	
А:&Ш!

_output_shapes
:	
А:"Щ

_output_shapes	
:А:"Ъ

_output_shapes	
:А:'Ы"
 
_output_shapes
:
А┘:'Ь"
 
_output_shapes
:
А┘:"Э

_output_shapes	
:┘:"Ю

_output_shapes	
:┘:Я

_output_shapes
: :а

_output_shapes
: :б

_output_shapes
: :в

_output_shapes
: :г

_output_shapes
: 
Ё
У
D__inference_conv1d_3_layer_call_and_return_conditional_losses_530075

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533570

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
п╕
╨
A__inference_model_layer_call_and_return_conditional_losses_531468

inputs(
layer_normalization_531227:(
layer_normalization_531229:0
kf_attention_cond_input_0:А0
kf_attention_cond_input_1:А0
kf_attention_cond_input_2:А'
kf_attention_cond_input_3:*
layer_normalization_1_531267:*
layer_normalization_1_531269:#
conv1d_531272:
conv1d_531274:%
conv1d_1_531278:
conv1d_1_531280:*
layer_normalization_2_531284:*
layer_normalization_2_531286:2
kf_attention_1_cond_input_0:А2
kf_attention_1_cond_input_1:А2
kf_attention_1_cond_input_2:А)
kf_attention_1_cond_input_3:*
layer_normalization_3_531324:*
layer_normalization_3_531326:%
conv1d_2_531329:
conv1d_2_531331:%
conv1d_3_531335:
conv1d_3_531337:*
layer_normalization_4_531341:*
layer_normalization_4_531343:2
kf_attention_2_cond_input_0:А2
kf_attention_2_cond_input_1:А2
kf_attention_2_cond_input_2:А)
kf_attention_2_cond_input_3:*
layer_normalization_5_531381:*
layer_normalization_5_531383:%
conv1d_4_531386:
conv1d_4_531388:%
conv1d_5_531392:
conv1d_5_531394:*
layer_normalization_6_531398:*
layer_normalization_6_531400:2
kf_attention_3_cond_input_0:А2
kf_attention_3_cond_input_1:А2
kf_attention_3_cond_input_2:А)
kf_attention_3_cond_input_3:*
layer_normalization_7_531438:*
layer_normalization_7_531440:%
conv1d_6_531443:
conv1d_6_531445:%
conv1d_7_531449:
conv1d_7_531451:
dense_531456:	
А
dense_531458:	А"
dense_1_531462:
А┘
dense_1_531464:	┘
identityИвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв"dropout_10/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallв"dropout_12/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв!dropout_5/StatefulPartitionedCallв!dropout_7/StatefulPartitionedCallв!dropout_8/StatefulPartitionedCallвkf_attention/condвkf_attention_1/condвkf_attention_2/condвkf_attention_3/condв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallв-layer_normalization_5/StatefulPartitionedCallв-layer_normalization_6/StatefulPartitionedCallв-layer_normalization_7/StatefulPartitionedCallа
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_531227layer_normalization_531229*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_529718}
kf_attention/cond/SqueezeSqueeze4layer_normalization/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:Т
kf_attention/condIf"kf_attention/cond/Squeeze:output:0kf_attention_cond_input_0kf_attention_cond_input_1kf_attention_cond_input_2kf_attention_cond_input_34layer_normalization/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*1
else_branch"R 
kf_attention_cond_false_531233**
output_shapes
:         
*0
then_branch!R
kf_attention_cond_true_531232x
kf_attention/cond/IdentityIdentitykf_attention/cond:output:0*
T0*+
_output_shapes
:         
ы
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#kf_attention/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_531073Н
tf.__operators__.add/AddV2AddV2*dropout_1/StatefulPartitionedCall:output:0inputs*
T0*+
_output_shapes
:         
└
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_531267layer_normalization_1_531269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_529832Ь
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_531272conv1d_531274*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_529854У
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_531030Ш
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_1_531278conv1d_1_531280*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_529882ж
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_531284layer_normalization_2_531286*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_529911Б
kf_attention_1/cond/SqueezeSqueeze6layer_normalization_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_1/condIf$kf_attention_1/cond/Squeeze:output:0kf_attention_1_cond_input_0kf_attention_1_cond_input_1kf_attention_1_cond_input_2kf_attention_1_cond_input_36layer_normalization_2/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_1_cond_false_531290**
output_shapes
:         
*2
then_branch#R!
kf_attention_1_cond_true_531289|
kf_attention_1/cond/IdentityIdentitykf_attention_1/cond:output:0*
T0*+
_output_shapes
:         
С
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall%kf_attention_1/cond/Identity:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_530960й
tf.__operators__.add_2/AddV2AddV2*dropout_4/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_531324layer_normalization_3_531326*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_530025д
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_531329conv1d_2_531331*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_530047Х
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_530917Ш
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0conv1d_3_531335conv1d_3_531337*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_530075и
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_531341layer_normalization_4_531343*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_530104Б
kf_attention_2/cond/SqueezeSqueeze6layer_normalization_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_2/condIf$kf_attention_2/cond/Squeeze:output:0kf_attention_2_cond_input_0kf_attention_2_cond_input_1kf_attention_2_cond_input_2kf_attention_2_cond_input_36layer_normalization_4/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_2_cond_false_531347**
output_shapes
:         
*2
then_branch#R!
kf_attention_2_cond_true_531346|
kf_attention_2/cond/IdentityIdentitykf_attention_2/cond:output:0*
T0*+
_output_shapes
:         
С
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall%kf_attention_2/cond/Identity:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_530847й
tf.__operators__.add_4/AddV2AddV2*dropout_7/StatefulPartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_531381layer_normalization_5_531383*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_530218д
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_531386conv1d_4_531388*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_530240Х
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_530804Ш
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv1d_5_531392conv1d_5_531394*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_530268и
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_531398layer_normalization_6_531400*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_530297Б
kf_attention_3/cond/SqueezeSqueeze6layer_normalization_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_3/condIf$kf_attention_3/cond/Squeeze:output:0kf_attention_3_cond_input_0kf_attention_3_cond_input_1kf_attention_3_cond_input_2kf_attention_3_cond_input_36layer_normalization_6/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_3_cond_false_531404**
output_shapes
:         
*2
then_branch#R!
kf_attention_3_cond_true_531403|
kf_attention_3/cond/IdentityIdentitykf_attention_3/cond:output:0*
T0*+
_output_shapes
:         
У
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall%kf_attention_3/cond/Identity:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_530734к
tf.__operators__.add_6/AddV2AddV2+dropout_10/StatefulPartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_531438layer_normalization_7_531440*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_530411д
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_531443conv1d_6_531445*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_530433Ш
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_530691Щ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0conv1d_7_531449conv1d_7_531451*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_530461и
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
Є
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_529686Р
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_531456dense_531458*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_530480Т
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_530648Т
dense_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_1_531462dense_1_531464*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_530504x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘╣	
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall^kf_attention/cond^kf_attention_1/cond^kf_attention_2/cond^kf_attention_3/cond,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2&
kf_attention/condkf_attention/cond2*
kf_attention_1/condkf_attention_1/cond2*
kf_attention_2/condkf_attention_2/cond2*
kf_attention_3/condkf_attention_3/cond2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_533607

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Б
c
*__inference_dropout_2_layer_call_fn_533450

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_531030s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_533455

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╚
У
D__inference_conv1d_2_layer_call_and_return_conditional_losses_533675

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_533372

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
и

e
F__inference_dropout_10_layer_call_and_return_conditional_losses_530734

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
│╕
╤
A__inference_model_layer_call_and_return_conditional_losses_532172
input_1(
layer_normalization_531931:(
layer_normalization_531933:0
kf_attention_cond_input_0:А0
kf_attention_cond_input_1:А0
kf_attention_cond_input_2:А'
kf_attention_cond_input_3:*
layer_normalization_1_531971:*
layer_normalization_1_531973:#
conv1d_531976:
conv1d_531978:%
conv1d_1_531982:
conv1d_1_531984:*
layer_normalization_2_531988:*
layer_normalization_2_531990:2
kf_attention_1_cond_input_0:А2
kf_attention_1_cond_input_1:А2
kf_attention_1_cond_input_2:А)
kf_attention_1_cond_input_3:*
layer_normalization_3_532028:*
layer_normalization_3_532030:%
conv1d_2_532033:
conv1d_2_532035:%
conv1d_3_532039:
conv1d_3_532041:*
layer_normalization_4_532045:*
layer_normalization_4_532047:2
kf_attention_2_cond_input_0:А2
kf_attention_2_cond_input_1:А2
kf_attention_2_cond_input_2:А)
kf_attention_2_cond_input_3:*
layer_normalization_5_532085:*
layer_normalization_5_532087:%
conv1d_4_532090:
conv1d_4_532092:%
conv1d_5_532096:
conv1d_5_532098:*
layer_normalization_6_532102:*
layer_normalization_6_532104:2
kf_attention_3_cond_input_0:А2
kf_attention_3_cond_input_1:А2
kf_attention_3_cond_input_2:А)
kf_attention_3_cond_input_3:*
layer_normalization_7_532142:*
layer_normalization_7_532144:%
conv1d_6_532147:
conv1d_6_532149:%
conv1d_7_532153:
conv1d_7_532155:
dense_532160:	
А
dense_532162:	А"
dense_1_532166:
А┘
dense_1_532168:	┘
identityИвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв"dropout_10/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallв"dropout_12/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв!dropout_5/StatefulPartitionedCallв!dropout_7/StatefulPartitionedCallв!dropout_8/StatefulPartitionedCallвkf_attention/condвkf_attention_1/condвkf_attention_2/condвkf_attention_3/condв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallв-layer_normalization_5/StatefulPartitionedCallв-layer_normalization_6/StatefulPartitionedCallв-layer_normalization_7/StatefulPartitionedCallб
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_531931layer_normalization_531933*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_529718}
kf_attention/cond/SqueezeSqueeze4layer_normalization/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:Т
kf_attention/condIf"kf_attention/cond/Squeeze:output:0kf_attention_cond_input_0kf_attention_cond_input_1kf_attention_cond_input_2kf_attention_cond_input_34layer_normalization/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*1
else_branch"R 
kf_attention_cond_false_531937**
output_shapes
:         
*0
then_branch!R
kf_attention_cond_true_531936x
kf_attention/cond/IdentityIdentitykf_attention/cond:output:0*
T0*+
_output_shapes
:         
ы
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#kf_attention/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_531073О
tf.__operators__.add/AddV2AddV2*dropout_1/StatefulPartitionedCall:output:0input_1*
T0*+
_output_shapes
:         
└
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_531971layer_normalization_1_531973*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_529832Ь
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_531976conv1d_531978*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_529854У
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_531030Ш
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_1_531982conv1d_1_531984*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_529882ж
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_531988layer_normalization_2_531990*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_529911Б
kf_attention_1/cond/SqueezeSqueeze6layer_normalization_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_1/condIf$kf_attention_1/cond/Squeeze:output:0kf_attention_1_cond_input_0kf_attention_1_cond_input_1kf_attention_1_cond_input_2kf_attention_1_cond_input_36layer_normalization_2/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_1_cond_false_531994**
output_shapes
:         
*2
then_branch#R!
kf_attention_1_cond_true_531993|
kf_attention_1/cond/IdentityIdentitykf_attention_1/cond:output:0*
T0*+
_output_shapes
:         
С
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall%kf_attention_1/cond/Identity:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_530960й
tf.__operators__.add_2/AddV2AddV2*dropout_4/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_532028layer_normalization_3_532030*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_530025д
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_532033conv1d_2_532035*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_530047Х
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_530917Ш
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0conv1d_3_532039conv1d_3_532041*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_530075и
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_532045layer_normalization_4_532047*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_530104Б
kf_attention_2/cond/SqueezeSqueeze6layer_normalization_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_2/condIf$kf_attention_2/cond/Squeeze:output:0kf_attention_2_cond_input_0kf_attention_2_cond_input_1kf_attention_2_cond_input_2kf_attention_2_cond_input_36layer_normalization_4/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_2_cond_false_532051**
output_shapes
:         
*2
then_branch#R!
kf_attention_2_cond_true_532050|
kf_attention_2/cond/IdentityIdentitykf_attention_2/cond:output:0*
T0*+
_output_shapes
:         
С
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall%kf_attention_2/cond/Identity:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_530847й
tf.__operators__.add_4/AddV2AddV2*dropout_7/StatefulPartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_532085layer_normalization_5_532087*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_530218д
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_532090conv1d_4_532092*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_530240Х
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_530804Ш
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv1d_5_532096conv1d_5_532098*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_530268и
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_532102layer_normalization_6_532104*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_530297Б
kf_attention_3/cond/SqueezeSqueeze6layer_normalization_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_3/condIf$kf_attention_3/cond/Squeeze:output:0kf_attention_3_cond_input_0kf_attention_3_cond_input_1kf_attention_3_cond_input_2kf_attention_3_cond_input_36layer_normalization_6/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_3_cond_false_532108**
output_shapes
:         
*2
then_branch#R!
kf_attention_3_cond_true_532107|
kf_attention_3/cond/IdentityIdentitykf_attention_3/cond:output:0*
T0*+
_output_shapes
:         
У
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall%kf_attention_3/cond/Identity:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_530734к
tf.__operators__.add_6/AddV2AddV2+dropout_10/StatefulPartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_532142layer_normalization_7_532144*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_530411д
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_532147conv1d_6_532149*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_530433Ш
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_530691Щ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0conv1d_7_532153conv1d_7_532155*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_530461и
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
Є
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_529686Р
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_532160dense_532162*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_530480Т
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_530648Т
dense_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_1_532166dense_1_532168*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_530504x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘╣	
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall^kf_attention/cond^kf_attention_1/cond^kf_attention_2/cond^kf_attention_3/cond,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2&
kf_attention/condkf_attention/cond2*
kf_attention_1/condkf_attention_1/cond2*
kf_attention_2/condkf_attention_2/cond2*
kf_attention_3/condkf_attention_3/cond2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:T P
+
_output_shapes
:         

!
_user_specified_name	input_1
п
F
*__inference_dropout_8_layer_call_fn_533915

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_530251d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_533757

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_530960

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╓
Ъ
)__inference_conv1d_1_layer_call_fn_533476

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_529882s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Эк
К
A__inference_model_layer_call_and_return_conditional_losses_531928
input_1(
layer_normalization_531687:(
layer_normalization_531689:0
kf_attention_cond_input_0:А0
kf_attention_cond_input_1:А0
kf_attention_cond_input_2:А'
kf_attention_cond_input_3:*
layer_normalization_1_531727:*
layer_normalization_1_531729:#
conv1d_531732:
conv1d_531734:%
conv1d_1_531738:
conv1d_1_531740:*
layer_normalization_2_531744:*
layer_normalization_2_531746:2
kf_attention_1_cond_input_0:А2
kf_attention_1_cond_input_1:А2
kf_attention_1_cond_input_2:А)
kf_attention_1_cond_input_3:*
layer_normalization_3_531784:*
layer_normalization_3_531786:%
conv1d_2_531789:
conv1d_2_531791:%
conv1d_3_531795:
conv1d_3_531797:*
layer_normalization_4_531801:*
layer_normalization_4_531803:2
kf_attention_2_cond_input_0:А2
kf_attention_2_cond_input_1:А2
kf_attention_2_cond_input_2:А)
kf_attention_2_cond_input_3:*
layer_normalization_5_531841:*
layer_normalization_5_531843:%
conv1d_4_531846:
conv1d_4_531848:%
conv1d_5_531852:
conv1d_5_531854:*
layer_normalization_6_531858:*
layer_normalization_6_531860:2
kf_attention_3_cond_input_0:А2
kf_attention_3_cond_input_1:А2
kf_attention_3_cond_input_2:А)
kf_attention_3_cond_input_3:*
layer_normalization_7_531898:*
layer_normalization_7_531900:%
conv1d_6_531903:
conv1d_6_531905:%
conv1d_7_531909:
conv1d_7_531911:
dense_531916:	
А
dense_531918:	А"
dense_1_531922:
А┘
dense_1_531924:	┘
identityИвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвkf_attention/condвkf_attention_1/condвkf_attention_2/condвkf_attention_3/condв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallв-layer_normalization_5/StatefulPartitionedCallв-layer_normalization_6/StatefulPartitionedCallв-layer_normalization_7/StatefulPartitionedCallб
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_531687layer_normalization_531689*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_529718}
kf_attention/cond/SqueezeSqueeze4layer_normalization/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:Т
kf_attention/condIf"kf_attention/cond/Squeeze:output:0kf_attention_cond_input_0kf_attention_cond_input_1kf_attention_cond_input_2kf_attention_cond_input_34layer_normalization/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*1
else_branch"R 
kf_attention_cond_false_531693**
output_shapes
:         
*0
then_branch!R
kf_attention_cond_true_531692x
kf_attention/cond/IdentityIdentitykf_attention/cond:output:0*
T0*+
_output_shapes
:         
█
dropout_1/PartitionedCallPartitionedCall#kf_attention/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_529807Ж
tf.__operators__.add/AddV2AddV2"dropout_1/PartitionedCall:output:0input_1*
T0*+
_output_shapes
:         
└
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_531727layer_normalization_1_531729*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_529832Ь
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_531732conv1d_531734*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_529854▀
dropout_2/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_529865Р
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_1_531738conv1d_1_531740*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_529882ж
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_531744layer_normalization_2_531746*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_529911Б
kf_attention_1/cond/SqueezeSqueeze6layer_normalization_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_1/condIf$kf_attention_1/cond/Squeeze:output:0kf_attention_1_cond_input_0kf_attention_1_cond_input_1kf_attention_1_cond_input_2kf_attention_1_cond_input_36layer_normalization_2/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_1_cond_false_531750**
output_shapes
:         
*2
then_branch#R!
kf_attention_1_cond_true_531749|
kf_attention_1/cond/IdentityIdentitykf_attention_1/cond:output:0*
T0*+
_output_shapes
:         
▌
dropout_4/PartitionedCallPartitionedCall%kf_attention_1/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_530000б
tf.__operators__.add_2/AddV2AddV2"dropout_4/PartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_531784layer_normalization_3_531786*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_530025д
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_531789conv1d_2_531791*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_530047с
dropout_5/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_530058Р
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0conv1d_3_531795conv1d_3_531797*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_530075и
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_531801layer_normalization_4_531803*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_530104Б
kf_attention_2/cond/SqueezeSqueeze6layer_normalization_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_2/condIf$kf_attention_2/cond/Squeeze:output:0kf_attention_2_cond_input_0kf_attention_2_cond_input_1kf_attention_2_cond_input_2kf_attention_2_cond_input_36layer_normalization_4/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_2_cond_false_531807**
output_shapes
:         
*2
then_branch#R!
kf_attention_2_cond_true_531806|
kf_attention_2/cond/IdentityIdentitykf_attention_2/cond:output:0*
T0*+
_output_shapes
:         
▌
dropout_7/PartitionedCallPartitionedCall%kf_attention_2/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_530193б
tf.__operators__.add_4/AddV2AddV2"dropout_7/PartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_531841layer_normalization_5_531843*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_530218д
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_531846conv1d_4_531848*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_530240с
dropout_8/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_530251Р
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv1d_5_531852conv1d_5_531854*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_530268и
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_531858layer_normalization_6_531860*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_530297Б
kf_attention_3/cond/SqueezeSqueeze6layer_normalization_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_3/condIf$kf_attention_3/cond/Squeeze:output:0kf_attention_3_cond_input_0kf_attention_3_cond_input_1kf_attention_3_cond_input_2kf_attention_3_cond_input_36layer_normalization_6/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_3_cond_false_531864**
output_shapes
:         
*2
then_branch#R!
kf_attention_3_cond_true_531863|
kf_attention_3/cond/IdentityIdentitykf_attention_3/cond:output:0*
T0*+
_output_shapes
:         
▀
dropout_10/PartitionedCallPartitionedCall%kf_attention_3/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_530386в
tf.__operators__.add_6/AddV2AddV2#dropout_10/PartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_531898layer_normalization_7_531900*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_530411д
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_531903conv1d_6_531905*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_530433у
dropout_11/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_530444С
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0conv1d_7_531909conv1d_7_531911*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_530461и
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
Є
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_529686Р
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_531916dense_531918*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_530480▌
dropout_12/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_530491К
dense_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_1_531922dense_1_531924*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_530504x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘Є
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^kf_attention/cond^kf_attention_1/cond^kf_attention_2/cond^kf_attention_3/cond,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2&
kf_attention/condkf_attention/cond2*
kf_attention_1/condkf_attention_1/cond2*
kf_attention_2/condkf_attention_2/cond2*
kf_attention_3/condkf_attention_3/cond2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:T P
+
_output_shapes
:         

!
_user_specified_name	input_1
М
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_529686

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
И
 
 kf_attention_3_cond_false_5318641
kf_attention_3_cond_531879:А1
kf_attention_3_cond_531881:А1
kf_attention_3_cond_531883:А(
kf_attention_3_cond_531885:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_531879kf_attention_3_cond_531881kf_attention_3_cond_531883kf_attention_3_cond_531885*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530363п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

И
 
 kf_attention_3_cond_false_5314041
kf_attention_3_cond_531419:А1
kf_attention_3_cond_531421:А1
kf_attention_3_cond_531423:А(
kf_attention_3_cond_531425:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_531419kf_attention_3_cond_531421kf_attention_3_cond_531423kf_attention_3_cond_531425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530363п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

Л
╞
H__inference_kf_attention_layer_call_and_return_conditional_losses_533357

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_1_cond_true_5319931
kf_attention_1_cond_531997:А1
kf_attention_1_cond_531999:А1
kf_attention_1_cond_532001:А(
kf_attention_1_cond_532003:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_531997kf_attention_1_cond_531999kf_attention_1_cond_532001kf_attention_1_cond_532003*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529943п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

З
■
kf_attention_3_cond_true_5321071
kf_attention_3_cond_532111:А1
kf_attention_3_cond_532113:А1
kf_attention_3_cond_532115:А(
kf_attention_3_cond_532117:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_532111kf_attention_3_cond_532113kf_attention_3_cond_532115kf_attention_3_cond_532117*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530329п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╘
э
kf_attention_cond_false_529725/
kf_attention_cond_529785:А/
kf_attention_cond_529787:А/
kf_attention_cond_529789:А&
kf_attention_cond_529791:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_529785kf_attention_cond_529787kf_attention_cond_529789kf_attention_cond_529791*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529784й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

З
■
kf_attention_2_cond_true_5318061
kf_attention_2_cond_531810:А1
kf_attention_2_cond_531812:А1
kf_attention_2_cond_531814:А(
kf_attention_2_cond_531816:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_531810kf_attention_2_cond_531812kf_attention_2_cond_531814kf_attention_2_cond_531816*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530136п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

Б
c
*__inference_dropout_1_layer_call_fn_533367

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_531073s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_529911

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534062

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_530218

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╘
э
kf_attention_cond_false_531693/
kf_attention_cond_531708:А/
kf_attention_cond_531710:А/
kf_attention_cond_531712:А&
kf_attention_cond_531714:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_531708kf_attention_cond_531710kf_attention_cond_531712kf_attention_cond_531714*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529784й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

└
Х
&__inference_dense_layer_call_fn_534216

inputs
unknown:	
А
	unknown_0:	А
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_530480p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
п
F
*__inference_dropout_7_layer_call_fn_533832

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_530193d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
п
F
*__inference_dropout_4_layer_call_fn_533597

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_530000d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529977

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_530251

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_531030

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530170

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
У
D__inference_conv1d_7_layer_call_and_return_conditional_losses_534196

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_533854

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ю
О
O__inference_layer_normalization_layer_call_and_return_conditional_losses_533287

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╘
э
kf_attention_cond_false_531233/
kf_attention_cond_531248:А/
kf_attention_cond_531250:А/
kf_attention_cond_531252:А&
kf_attention_cond_531254:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_531248kf_attention_cond_531250kf_attention_cond_531252kf_attention_cond_531254*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529784й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╙
ь
kf_attention_cond_true_529724/
kf_attention_cond_529751:А/
kf_attention_cond_529753:А/
kf_attention_cond_529755:А&
kf_attention_cond_529757:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_529751kf_attention_cond_529753kf_attention_cond_529755kf_attention_cond_529757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529750й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

ш
Я
6__inference_layer_normalization_2_layer_call_fn_533500

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_529911s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_1_cond_true_5312891
kf_attention_1_cond_531293:А1
kf_attention_1_cond_531295:А1
kf_attention_1_cond_531297:А(
kf_attention_1_cond_531299:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_531293kf_attention_1_cond_531295kf_attention_1_cond_531297kf_attention_1_cond_531299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529943п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

З
■
kf_attention_3_cond_true_5303031
kf_attention_3_cond_530330:А1
kf_attention_3_cond_530332:А1
kf_attention_3_cond_530334:А(
kf_attention_3_cond_530336:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_530330kf_attention_3_cond_530332kf_attention_3_cond_530334kf_attention_3_cond_530336*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530329п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

И
 
 kf_attention_2_cond_false_5313471
kf_attention_2_cond_531362:А1
kf_attention_2_cond_531364:А1
kf_attention_2_cond_531366:А(
kf_attention_2_cond_531368:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_531362kf_attention_2_cond_531364kf_attention_2_cond_531366kf_attention_2_cond_531368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530170п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

╘
э
kf_attention_cond_false_531937/
kf_attention_cond_531952:А/
kf_attention_cond_531954:А/
kf_attention_cond_531956:А&
kf_attention_cond_531958:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_531952kf_attention_cond_531954kf_attention_cond_531956kf_attention_cond_531958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529784й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

и

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_530691

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
щ
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_530386

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Я
╙
&__inference_model_layer_call_fn_532394

inputs
unknown:
	unknown_0: 
	unknown_1:А 
	unknown_2:А 
	unknown_3:А
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:!

unknown_13:А!

unknown_14:А!

unknown_15:А

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:!

unknown_25:А!

unknown_26:А!

unknown_27:А

unknown_28:

unknown_29:

unknown_30: 

unknown_31:

unknown_32: 

unknown_33:

unknown_34:

unknown_35:

unknown_36:!

unknown_37:А!

unknown_38:А!

unknown_39:А

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:	
А

unknown_48:	А

unknown_49:
А┘

unknown_50:	┘
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_530511p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╙
ь
kf_attention_cond_true_531936/
kf_attention_cond_531940:А/
kf_attention_cond_531942:А/
kf_attention_cond_531944:А&
kf_attention_cond_531946:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_531940kf_attention_cond_531942kf_attention_cond_531944kf_attention_cond_531946*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529750й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

Ё
У
D__inference_conv1d_1_layer_call_and_return_conditional_losses_529882

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_2_cond_true_5320501
kf_attention_2_cond_532054:А1
kf_attention_2_cond_532056:А1
kf_attention_2_cond_532058:А(
kf_attention_2_cond_532060:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_532054kf_attention_2_cond_532056kf_attention_2_cond_532058kf_attention_2_cond_532060*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530136п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

щ
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_534160

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╟
Ш
(__inference_dense_1_layer_call_fn_534263

inputs
unknown:
А┘
	unknown_0:	┘
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_530504p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
А
╥
$__inference_signature_wrapper_532285
input_1
unknown:
	unknown_0: 
	unknown_1:А 
	unknown_2:А 
	unknown_3:А
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:!

unknown_13:А!

unknown_14:А!

unknown_15:А

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:!

unknown_25:А!

unknown_26:А!

unknown_27:А

unknown_28:

unknown_29:

unknown_30: 

unknown_31:

unknown_32: 

unknown_33:

unknown_34:

unknown_35:

unknown_36:!

unknown_37:А!

unknown_38:А!

unknown_39:А

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:	
А

unknown_48:	А

unknown_49:
А┘

unknown_50:	┘
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_529676p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         

!
_user_specified_name	input_1
з

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_530847

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
щ
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_534077

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_531073

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_533522

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ў
d
+__inference_dropout_12_layer_call_fn_534237

inputs
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_530648p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ё
У
D__inference_conv1d_5_layer_call_and_return_conditional_losses_533961

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
е
G
+__inference_dropout_12_layer_call_fn_534232

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_530491a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
з

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_533937

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_533842

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┌Ж
о3
!__inference__wrapped_model_529676
input_1M
?model_layer_normalization_batchnorm_mul_readvariableop_resource:I
;model_layer_normalization_batchnorm_readvariableop_resource:O
8model_kf_attention_einsum_einsum_readvariableop_resource:АQ
:model_kf_attention_einsum_1_einsum_readvariableop_resource:АQ
:model_kf_attention_einsum_4_einsum_readvariableop_resource:А<
.model_kf_attention_add_readvariableop_resource:O
Amodel_layer_normalization_1_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_1_batchnorm_readvariableop_resource:N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource::
,model_conv1d_biasadd_readvariableop_resource:P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_1_biasadd_readvariableop_resource:O
Amodel_layer_normalization_2_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_2_batchnorm_readvariableop_resource:Q
:model_kf_attention_1_einsum_einsum_readvariableop_resource:АS
<model_kf_attention_1_einsum_1_einsum_readvariableop_resource:АS
<model_kf_attention_1_einsum_4_einsum_readvariableop_resource:А>
0model_kf_attention_1_add_readvariableop_resource:O
Amodel_layer_normalization_3_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_3_batchnorm_readvariableop_resource:P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_2_biasadd_readvariableop_resource:P
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_3_biasadd_readvariableop_resource:O
Amodel_layer_normalization_4_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_4_batchnorm_readvariableop_resource:Q
:model_kf_attention_2_einsum_einsum_readvariableop_resource:АS
<model_kf_attention_2_einsum_1_einsum_readvariableop_resource:АS
<model_kf_attention_2_einsum_4_einsum_readvariableop_resource:А>
0model_kf_attention_2_add_readvariableop_resource:O
Amodel_layer_normalization_5_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_5_batchnorm_readvariableop_resource:P
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_4_biasadd_readvariableop_resource:P
:model_conv1d_5_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_5_biasadd_readvariableop_resource:O
Amodel_layer_normalization_6_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_6_batchnorm_readvariableop_resource:Q
:model_kf_attention_3_einsum_einsum_readvariableop_resource:АS
<model_kf_attention_3_einsum_1_einsum_readvariableop_resource:АS
<model_kf_attention_3_einsum_4_einsum_readvariableop_resource:А>
0model_kf_attention_3_add_readvariableop_resource:O
Amodel_layer_normalization_7_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_7_batchnorm_readvariableop_resource:P
:model_conv1d_6_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_6_biasadd_readvariableop_resource:P
:model_conv1d_7_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_7_biasadd_readvariableop_resource:=
*model_dense_matmul_readvariableop_resource:	
А:
+model_dense_biasadd_readvariableop_resource:	А@
,model_dense_1_matmul_readvariableop_resource:
А┘<
-model_dense_1_biasadd_readvariableop_resource:	┘
identityИв#model/conv1d/BiasAdd/ReadVariableOpв/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_1/BiasAdd/ReadVariableOpв1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_2/BiasAdd/ReadVariableOpв1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_3/BiasAdd/ReadVariableOpв1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_4/BiasAdd/ReadVariableOpв1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_5/BiasAdd/ReadVariableOpв1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_6/BiasAdd/ReadVariableOpв1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpв%model/conv1d_7/BiasAdd/ReadVariableOpв1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв%model/kf_attention/add/ReadVariableOpв/model/kf_attention/einsum/Einsum/ReadVariableOpв1model/kf_attention/einsum_1/Einsum/ReadVariableOpв1model/kf_attention/einsum_4/Einsum/ReadVariableOpв'model/kf_attention_1/add/ReadVariableOpв1model/kf_attention_1/einsum/Einsum/ReadVariableOpв3model/kf_attention_1/einsum_1/Einsum/ReadVariableOpв3model/kf_attention_1/einsum_4/Einsum/ReadVariableOpв'model/kf_attention_2/add/ReadVariableOpв1model/kf_attention_2/einsum/Einsum/ReadVariableOpв3model/kf_attention_2/einsum_1/Einsum/ReadVariableOpв3model/kf_attention_2/einsum_4/Einsum/ReadVariableOpв'model/kf_attention_3/add/ReadVariableOpв1model/kf_attention_3/einsum/Einsum/ReadVariableOpв3model/kf_attention_3/einsum_1/Einsum/ReadVariableOpв3model/kf_attention_3/einsum_4/Einsum/ReadVariableOpв2model/layer_normalization/batchnorm/ReadVariableOpв6model/layer_normalization/batchnorm/mul/ReadVariableOpв4model/layer_normalization_1/batchnorm/ReadVariableOpв8model/layer_normalization_1/batchnorm/mul/ReadVariableOpв4model/layer_normalization_2/batchnorm/ReadVariableOpв8model/layer_normalization_2/batchnorm/mul/ReadVariableOpв4model/layer_normalization_3/batchnorm/ReadVariableOpв8model/layer_normalization_3/batchnorm/mul/ReadVariableOpв4model/layer_normalization_4/batchnorm/ReadVariableOpв8model/layer_normalization_4/batchnorm/mul/ReadVariableOpв4model/layer_normalization_5/batchnorm/ReadVariableOpв8model/layer_normalization_5/batchnorm/mul/ReadVariableOpв4model/layer_normalization_6/batchnorm/ReadVariableOpв8model/layer_normalization_6/batchnorm/mul/ReadVariableOpв4model/layer_normalization_7/batchnorm/ReadVariableOpв8model/layer_normalization_7/batchnorm/mul/ReadVariableOpВ
8model/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┴
&model/layer_normalization/moments/meanMeaninput_1Amodel/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(е
.model/layer_normalization/moments/StopGradientStopGradient/model/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         
└
3model/layer_normalization/moments/SquaredDifferenceSquaredDifferenceinput_17model/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
Ж
<model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:∙
*model/layer_normalization/moments/varianceMean7model/layer_normalization/moments/SquaredDifference:z:0Emodel/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(n
)model/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╧
'model/layer_normalization/batchnorm/addAddV23model/layer_normalization/moments/variance:output:02model/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Х
)model/layer_normalization/batchnorm/RsqrtRsqrt+model/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         
▓
6model/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╙
'model/layer_normalization/batchnorm/mulMul-model/layer_normalization/batchnorm/Rsqrt:y:0>model/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ь
)model/layer_normalization/batchnorm/mul_1Mulinput_1+model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
─
)model/layer_normalization/batchnorm/mul_2Mul/model/layer_normalization/moments/mean:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
к
2model/layer_normalization/batchnorm/ReadVariableOpReadVariableOp;model_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╧
'model/layer_normalization/batchnorm/subSub:model/layer_normalization/batchnorm/ReadVariableOp:value:0-model/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
─
)model/layer_normalization/batchnorm/add_1AddV2-model/layer_normalization/batchnorm/mul_1:z:0+model/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
н
/model/kf_attention/einsum/Einsum/ReadVariableOpReadVariableOp8model_kf_attention_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ў
 model/kf_attention/einsum/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:07model/kf_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHO▒
1model/kf_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp:model_kf_attention_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0√
"model/kf_attention/einsum_1/EinsumEinsum-model/layer_normalization/batchnorm/add_1:z:09model/kf_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHO]
model/kf_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АCc
model/kf_attention/SqrtSqrt!model/kf_attention/Const:output:0*
T0*
_output_shapes
: и
model/kf_attention/truedivRealDiv)model/kf_attention/einsum/Einsum:output:0model/kf_attention/Sqrt:y:0*
T0*0
_output_shapes
:         
Ас
"model/kf_attention/einsum_2/EinsumEinsummodel/kf_attention/truediv:z:0+model/kf_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMМ
model/kf_attention/SoftmaxSoftmax+model/kf_attention/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

ш
"model/kf_attention/einsum_3/EinsumEinsum$model/kf_attention/Softmax:softmax:0+model/kf_attention/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHI▒
1model/kf_attention/einsum_4/Einsum/ReadVariableOpReadVariableOp:model_kf_attention_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0Ї
"model/kf_attention/einsum_4/EinsumEinsum+model/kf_attention/einsum_3/Einsum:output:09model/kf_attention/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOР
%model/kf_attention/add/ReadVariableOpReadVariableOp.model_kf_attention_add_readvariableop_resource*
_output_shapes
:*
dtype0▒
model/kf_attention/addAddV2+model/kf_attention/einsum_4/Einsum:output:0-model/kf_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
v
model/dropout_1/IdentityIdentitymodel/kf_attention/add:z:0*
T0*+
_output_shapes
:         
Л
 model/tf.__operators__.add/AddV2AddV2!model/dropout_1/Identity:output:0input_1*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:т
(model/layer_normalization_1/moments/meanMean$model/tf.__operators__.add/AddV2:z:0Cmodel/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_1/moments/StopGradientStopGradient1model/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         
с
5model/layer_normalization_1/moments/SquaredDifferenceSquaredDifference$model/tf.__operators__.add/AddV2:z:09model/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_1/moments/varianceMean9model/layer_normalization_1/moments/SquaredDifference:z:0Gmodel/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_1/batchnorm/addAddV25model/layer_normalization_1/moments/variance:output:04model/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_1/batchnorm/RsqrtRsqrt-model/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_1/batchnorm/mulMul/model/layer_normalization_1/batchnorm/Rsqrt:y:0@model/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╜
+model/layer_normalization_1/batchnorm/mul_1Mul$model/tf.__operators__.add/AddV2:z:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_1/batchnorm/mul_2Mul1model/layer_normalization_1/moments/mean:output:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_1/batchnorm/subSub<model/layer_normalization_1/batchnorm/ReadVariableOp:value:0/model/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_1/batchnorm/add_1AddV2/model/layer_normalization_1/batchnorm/mul_1:z:0-model/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
m
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ─
model/conv1d/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_1/batchnorm/add_1:z:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
м
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╟
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╘
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ъ
model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        М
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0и
model/conv1d/BiasAddBiasAdd$model/conv1d/Conv1D/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
n
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:         
{
model/dropout_2/IdentityIdentitymodel/conv1d/Relu:activations:0*
T0*+
_output_shapes
:         
o
$model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ║
 model/conv1d_1/Conv1D/ExpandDims
ExpandDims!model/dropout_2/Identity:output:0-model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_1/Conv1DConv2D)model/conv1d_1/Conv1D/ExpandDims:output:0+model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_1/Conv1D/SqueezeSqueezemodel/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/Conv1D/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
"model/tf.__operators__.add_1/AddV2AddV2model/conv1d_1/BiasAdd:output:0$model/tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ф
(model/layer_normalization_2/moments/meanMean&model/tf.__operators__.add_1/AddV2:z:0Cmodel/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_2/moments/StopGradientStopGradient1model/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:         
у
5model/layer_normalization_2/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_1/AddV2:z:09model/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_2/moments/varianceMean9model/layer_normalization_2/moments/SquaredDifference:z:0Gmodel/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_2/batchnorm/addAddV25model/layer_normalization_2/moments/variance:output:04model/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_2/batchnorm/RsqrtRsqrt-model/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_2/batchnorm/mulMul/model/layer_normalization_2/batchnorm/Rsqrt:y:0@model/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┐
+model/layer_normalization_2/batchnorm/mul_1Mul&model/tf.__operators__.add_1/AddV2:z:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_2/batchnorm/mul_2Mul1model/layer_normalization_2/moments/mean:output:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_2/batchnorm/subSub<model/layer_normalization_2/batchnorm/ReadVariableOp:value:0/model/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_2/batchnorm/add_1AddV2/model/layer_normalization_2/batchnorm/mul_1:z:0-model/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
▒
1model/kf_attention_1/einsum/Einsum/ReadVariableOpReadVariableOp:model_kf_attention_1_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0¤
"model/kf_attention_1/einsum/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:09model/kf_attention_1/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHO╡
3model/kf_attention_1/einsum_1/Einsum/ReadVariableOpReadVariableOp<model_kf_attention_1_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0Б
$model/kf_attention_1/einsum_1/EinsumEinsum/model/layer_normalization_2/batchnorm/add_1:z:0;model/kf_attention_1/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHO_
model/kf_attention_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АCg
model/kf_attention_1/SqrtSqrt#model/kf_attention_1/Const:output:0*
T0*
_output_shapes
: о
model/kf_attention_1/truedivRealDiv+model/kf_attention_1/einsum/Einsum:output:0model/kf_attention_1/Sqrt:y:0*
T0*0
_output_shapes
:         
Ач
$model/kf_attention_1/einsum_2/EinsumEinsum model/kf_attention_1/truediv:z:0-model/kf_attention_1/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMР
model/kf_attention_1/SoftmaxSoftmax-model/kf_attention_1/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

ю
$model/kf_attention_1/einsum_3/EinsumEinsum&model/kf_attention_1/Softmax:softmax:0-model/kf_attention_1/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHI╡
3model/kf_attention_1/einsum_4/Einsum/ReadVariableOpReadVariableOp<model_kf_attention_1_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0·
$model/kf_attention_1/einsum_4/EinsumEinsum-model/kf_attention_1/einsum_3/Einsum:output:0;model/kf_attention_1/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOФ
'model/kf_attention_1/add/ReadVariableOpReadVariableOp0model_kf_attention_1_add_readvariableop_resource*
_output_shapes
:*
dtype0╖
model/kf_attention_1/addAddV2-model/kf_attention_1/einsum_4/Einsum:output:0/model/kf_attention_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
x
model/dropout_4/IdentityIdentitymodel/kf_attention_1/add:z:0*
T0*+
_output_shapes
:         
м
"model/tf.__operators__.add_2/AddV2AddV2!model/dropout_4/Identity:output:0&model/tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ф
(model/layer_normalization_3/moments/meanMean&model/tf.__operators__.add_2/AddV2:z:0Cmodel/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_3/moments/StopGradientStopGradient1model/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:         
у
5model/layer_normalization_3/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_2/AddV2:z:09model/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_3/moments/varianceMean9model/layer_normalization_3/moments/SquaredDifference:z:0Gmodel/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_3/batchnorm/addAddV25model/layer_normalization_3/moments/variance:output:04model/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_3/batchnorm/RsqrtRsqrt-model/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_3/batchnorm/mulMul/model/layer_normalization_3/batchnorm/Rsqrt:y:0@model/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┐
+model/layer_normalization_3/batchnorm/mul_1Mul&model/tf.__operators__.add_2/AddV2:z:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_3/batchnorm/mul_2Mul1model/layer_normalization_3/moments/mean:output:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_3/batchnorm/subSub<model/layer_normalization_3/batchnorm/ReadVariableOp:value:0/model/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_3/batchnorm/add_1AddV2/model/layer_normalization_3/batchnorm/mul_1:z:0-model/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
o
$model/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╚
 model/conv1d_2/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_3/batchnorm/add_1:z:0-model/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_2/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_2/Conv1DConv2D)model/conv1d_2/Conv1D/ExpandDims:output:0+model/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_2/Conv1D/SqueezeSqueezemodel/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/Conv1D/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
model/conv1d_2/ReluRelumodel/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:         
}
model/dropout_5/IdentityIdentity!model/conv1d_2/Relu:activations:0*
T0*+
_output_shapes
:         
o
$model/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ║
 model/conv1d_3/Conv1D/ExpandDims
ExpandDims!model/dropout_5/Identity:output:0-model/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_3/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_3/Conv1DConv2D)model/conv1d_3/Conv1D/ExpandDims:output:0+model/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_3/Conv1D/SqueezeSqueezemodel/conv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_3/BiasAddBiasAdd&model/conv1d_3/Conv1D/Squeeze:output:0-model/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
к
"model/tf.__operators__.add_3/AddV2AddV2model/conv1d_3/BiasAdd:output:0&model/tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ф
(model/layer_normalization_4/moments/meanMean&model/tf.__operators__.add_3/AddV2:z:0Cmodel/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_4/moments/StopGradientStopGradient1model/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:         
у
5model/layer_normalization_4/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_3/AddV2:z:09model/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_4/moments/varianceMean9model/layer_normalization_4/moments/SquaredDifference:z:0Gmodel/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_4/batchnorm/addAddV25model/layer_normalization_4/moments/variance:output:04model/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_4/batchnorm/RsqrtRsqrt-model/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_4/batchnorm/mulMul/model/layer_normalization_4/batchnorm/Rsqrt:y:0@model/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┐
+model/layer_normalization_4/batchnorm/mul_1Mul&model/tf.__operators__.add_3/AddV2:z:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_4/batchnorm/mul_2Mul1model/layer_normalization_4/moments/mean:output:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_4/batchnorm/subSub<model/layer_normalization_4/batchnorm/ReadVariableOp:value:0/model/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_4/batchnorm/add_1AddV2/model/layer_normalization_4/batchnorm/mul_1:z:0-model/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
▒
1model/kf_attention_2/einsum/Einsum/ReadVariableOpReadVariableOp:model_kf_attention_2_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0¤
"model/kf_attention_2/einsum/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:09model/kf_attention_2/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHO╡
3model/kf_attention_2/einsum_1/Einsum/ReadVariableOpReadVariableOp<model_kf_attention_2_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0Б
$model/kf_attention_2/einsum_1/EinsumEinsum/model/layer_normalization_4/batchnorm/add_1:z:0;model/kf_attention_2/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHO_
model/kf_attention_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АCg
model/kf_attention_2/SqrtSqrt#model/kf_attention_2/Const:output:0*
T0*
_output_shapes
: о
model/kf_attention_2/truedivRealDiv+model/kf_attention_2/einsum/Einsum:output:0model/kf_attention_2/Sqrt:y:0*
T0*0
_output_shapes
:         
Ач
$model/kf_attention_2/einsum_2/EinsumEinsum model/kf_attention_2/truediv:z:0-model/kf_attention_2/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMР
model/kf_attention_2/SoftmaxSoftmax-model/kf_attention_2/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

ю
$model/kf_attention_2/einsum_3/EinsumEinsum&model/kf_attention_2/Softmax:softmax:0-model/kf_attention_2/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHI╡
3model/kf_attention_2/einsum_4/Einsum/ReadVariableOpReadVariableOp<model_kf_attention_2_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0·
$model/kf_attention_2/einsum_4/EinsumEinsum-model/kf_attention_2/einsum_3/Einsum:output:0;model/kf_attention_2/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOФ
'model/kf_attention_2/add/ReadVariableOpReadVariableOp0model_kf_attention_2_add_readvariableop_resource*
_output_shapes
:*
dtype0╖
model/kf_attention_2/addAddV2-model/kf_attention_2/einsum_4/Einsum:output:0/model/kf_attention_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
x
model/dropout_7/IdentityIdentitymodel/kf_attention_2/add:z:0*
T0*+
_output_shapes
:         
м
"model/tf.__operators__.add_4/AddV2AddV2!model/dropout_7/Identity:output:0&model/tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ф
(model/layer_normalization_5/moments/meanMean&model/tf.__operators__.add_4/AddV2:z:0Cmodel/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_5/moments/StopGradientStopGradient1model/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:         
у
5model/layer_normalization_5/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_4/AddV2:z:09model/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_5/moments/varianceMean9model/layer_normalization_5/moments/SquaredDifference:z:0Gmodel/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_5/batchnorm/addAddV25model/layer_normalization_5/moments/variance:output:04model/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_5/batchnorm/RsqrtRsqrt-model/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_5/batchnorm/mulMul/model/layer_normalization_5/batchnorm/Rsqrt:y:0@model/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┐
+model/layer_normalization_5/batchnorm/mul_1Mul&model/tf.__operators__.add_4/AddV2:z:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_5/batchnorm/mul_2Mul1model/layer_normalization_5/moments/mean:output:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_5/batchnorm/subSub<model/layer_normalization_5/batchnorm/ReadVariableOp:value:0/model/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_5/batchnorm/add_1AddV2/model/layer_normalization_5/batchnorm/mul_1:z:0-model/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
o
$model/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╚
 model/conv1d_4/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_5/batchnorm/add_1:z:0-model/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_4/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_4/Conv1DConv2D)model/conv1d_4/Conv1D/ExpandDims:output:0+model/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_4/Conv1D/SqueezeSqueezemodel/conv1d_4/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_4/BiasAddBiasAdd&model/conv1d_4/Conv1D/Squeeze:output:0-model/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
model/conv1d_4/ReluRelumodel/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:         
}
model/dropout_8/IdentityIdentity!model/conv1d_4/Relu:activations:0*
T0*+
_output_shapes
:         
o
$model/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ║
 model/conv1d_5/Conv1D/ExpandDims
ExpandDims!model/dropout_8/Identity:output:0-model/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_5/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_5/Conv1DConv2D)model/conv1d_5/Conv1D/ExpandDims:output:0+model/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_5/Conv1D/SqueezeSqueezemodel/conv1d_5/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_5/BiasAddBiasAdd&model/conv1d_5/Conv1D/Squeeze:output:0-model/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
к
"model/tf.__operators__.add_5/AddV2AddV2model/conv1d_5/BiasAdd:output:0&model/tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ф
(model/layer_normalization_6/moments/meanMean&model/tf.__operators__.add_5/AddV2:z:0Cmodel/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_6/moments/StopGradientStopGradient1model/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:         
у
5model/layer_normalization_6/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_5/AddV2:z:09model/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_6/moments/varianceMean9model/layer_normalization_6/moments/SquaredDifference:z:0Gmodel/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_6/batchnorm/addAddV25model/layer_normalization_6/moments/variance:output:04model/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_6/batchnorm/RsqrtRsqrt-model/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_6/batchnorm/mulMul/model/layer_normalization_6/batchnorm/Rsqrt:y:0@model/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┐
+model/layer_normalization_6/batchnorm/mul_1Mul&model/tf.__operators__.add_5/AddV2:z:0-model/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_6/batchnorm/mul_2Mul1model/layer_normalization_6/moments/mean:output:0-model/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_6/batchnorm/subSub<model/layer_normalization_6/batchnorm/ReadVariableOp:value:0/model/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_6/batchnorm/add_1AddV2/model/layer_normalization_6/batchnorm/mul_1:z:0-model/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
▒
1model/kf_attention_3/einsum/Einsum/ReadVariableOpReadVariableOp:model_kf_attention_3_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0¤
"model/kf_attention_3/einsum/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:09model/kf_attention_3/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHO╡
3model/kf_attention_3/einsum_1/Einsum/ReadVariableOpReadVariableOp<model_kf_attention_3_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0Б
$model/kf_attention_3/einsum_1/EinsumEinsum/model/layer_normalization_6/batchnorm/add_1:z:0;model/kf_attention_3/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHO_
model/kf_attention_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АCg
model/kf_attention_3/SqrtSqrt#model/kf_attention_3/Const:output:0*
T0*
_output_shapes
: о
model/kf_attention_3/truedivRealDiv+model/kf_attention_3/einsum/Einsum:output:0model/kf_attention_3/Sqrt:y:0*
T0*0
_output_shapes
:         
Ач
$model/kf_attention_3/einsum_2/EinsumEinsum model/kf_attention_3/truediv:z:0-model/kf_attention_3/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMР
model/kf_attention_3/SoftmaxSoftmax-model/kf_attention_3/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

ю
$model/kf_attention_3/einsum_3/EinsumEinsum&model/kf_attention_3/Softmax:softmax:0-model/kf_attention_3/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHI╡
3model/kf_attention_3/einsum_4/Einsum/ReadVariableOpReadVariableOp<model_kf_attention_3_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0·
$model/kf_attention_3/einsum_4/EinsumEinsum-model/kf_attention_3/einsum_3/Einsum:output:0;model/kf_attention_3/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOФ
'model/kf_attention_3/add/ReadVariableOpReadVariableOp0model_kf_attention_3_add_readvariableop_resource*
_output_shapes
:*
dtype0╖
model/kf_attention_3/addAddV2-model/kf_attention_3/einsum_4/Einsum:output:0/model/kf_attention_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
y
model/dropout_10/IdentityIdentitymodel/kf_attention_3/add:z:0*
T0*+
_output_shapes
:         
н
"model/tf.__operators__.add_6/AddV2AddV2"model/dropout_10/Identity:output:0&model/tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
Д
:model/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ф
(model/layer_normalization_7/moments/meanMean&model/tf.__operators__.add_6/AddV2:z:0Cmodel/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(й
0model/layer_normalization_7/moments/StopGradientStopGradient1model/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:         
у
5model/layer_normalization_7/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_6/AddV2:z:09model/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
И
>model/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
,model/layer_normalization_7/moments/varianceMean9model/layer_normalization_7/moments/SquaredDifference:z:0Gmodel/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(p
+model/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╒
)model/layer_normalization_7/batchnorm/addAddV25model/layer_normalization_7/moments/variance:output:04model/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Щ
+model/layer_normalization_7/batchnorm/RsqrtRsqrt-model/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:         
╢
8model/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┘
)model/layer_normalization_7/batchnorm/mulMul/model/layer_normalization_7/batchnorm/Rsqrt:y:0@model/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┐
+model/layer_normalization_7/batchnorm/mul_1Mul&model/tf.__operators__.add_6/AddV2:z:0-model/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_7/batchnorm/mul_2Mul1model/layer_normalization_7/moments/mean:output:0-model/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
о
4model/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╒
)model/layer_normalization_7/batchnorm/subSub<model/layer_normalization_7/batchnorm/ReadVariableOp:value:0/model/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╩
+model/layer_normalization_7/batchnorm/add_1AddV2/model/layer_normalization_7/batchnorm/mul_1:z:0-model/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
o
$model/conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╚
 model/conv1d_6/Conv1D/ExpandDims
ExpandDims/model/layer_normalization_7/batchnorm/add_1:z:0-model/conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_6/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_6/Conv1DConv2D)model/conv1d_6/Conv1D/ExpandDims:output:0+model/conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_6/Conv1D/SqueezeSqueezemodel/conv1d_6/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_6/BiasAddBiasAdd&model/conv1d_6/Conv1D/Squeeze:output:0-model/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
model/conv1d_6/ReluRelumodel/conv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:         
~
model/dropout_11/IdentityIdentity!model/conv1d_6/Relu:activations:0*
T0*+
_output_shapes
:         
o
$model/conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╗
 model/conv1d_7/Conv1D/ExpandDims
ExpandDims"model/dropout_11/Identity:output:0-model/conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
░
1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0h
&model/conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ═
"model/conv1d_7/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┌
model/conv1d_7/Conv1DConv2D)model/conv1d_7/Conv1D/ExpandDims:output:0+model/conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Ю
model/conv1d_7/Conv1D/SqueezeSqueezemodel/conv1d_7/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Р
%model/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
model/conv1d_7/BiasAddBiasAdd&model/conv1d_7/Conv1D/Squeeze:output:0-model/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
к
"model/tf.__operators__.add_7/AddV2AddV2model/conv1d_7/BiasAdd:output:0&model/tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :┼
#model/global_average_pooling1d/MeanMean&model/tf.__operators__.add_7/AddV2:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         
Н
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	
А*
dtype0и
model/dense/MatMulMatMul,model/global_average_pooling1d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЛ
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ы
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Аx
model/dropout_12/IdentityIdentitymodel/dense/Relu:activations:0*
T0*(
_output_shapes
:         АТ
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
А┘*
dtype0в
model/dense_1/MatMulMatMul"model/dropout_12/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘П
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:┘*
dtype0б
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘s
model/dense_1/SoftmaxSoftmaxmodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ┘o
IdentityIdentitymodel/dense_1/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ┘ф
NoOpNoOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_2/BiasAdd/ReadVariableOp2^model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_3/BiasAdd/ReadVariableOp2^model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_4/BiasAdd/ReadVariableOp2^model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_5/BiasAdd/ReadVariableOp2^model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_6/BiasAdd/ReadVariableOp2^model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_7/BiasAdd/ReadVariableOp2^model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp&^model/kf_attention/add/ReadVariableOp0^model/kf_attention/einsum/Einsum/ReadVariableOp2^model/kf_attention/einsum_1/Einsum/ReadVariableOp2^model/kf_attention/einsum_4/Einsum/ReadVariableOp(^model/kf_attention_1/add/ReadVariableOp2^model/kf_attention_1/einsum/Einsum/ReadVariableOp4^model/kf_attention_1/einsum_1/Einsum/ReadVariableOp4^model/kf_attention_1/einsum_4/Einsum/ReadVariableOp(^model/kf_attention_2/add/ReadVariableOp2^model/kf_attention_2/einsum/Einsum/ReadVariableOp4^model/kf_attention_2/einsum_1/Einsum/ReadVariableOp4^model/kf_attention_2/einsum_4/Einsum/ReadVariableOp(^model/kf_attention_3/add/ReadVariableOp2^model/kf_attention_3/einsum/Einsum/ReadVariableOp4^model/kf_attention_3/einsum_1/Einsum/ReadVariableOp4^model/kf_attention_3/einsum_4/Einsum/ReadVariableOp3^model/layer_normalization/batchnorm/ReadVariableOp7^model/layer_normalization/batchnorm/mul/ReadVariableOp5^model/layer_normalization_1/batchnorm/ReadVariableOp9^model/layer_normalization_1/batchnorm/mul/ReadVariableOp5^model/layer_normalization_2/batchnorm/ReadVariableOp9^model/layer_normalization_2/batchnorm/mul/ReadVariableOp5^model/layer_normalization_3/batchnorm/ReadVariableOp9^model/layer_normalization_3/batchnorm/mul/ReadVariableOp5^model/layer_normalization_4/batchnorm/ReadVariableOp9^model/layer_normalization_4/batchnorm/mul/ReadVariableOp5^model/layer_normalization_5/batchnorm/ReadVariableOp9^model/layer_normalization_5/batchnorm/mul/ReadVariableOp5^model/layer_normalization_6/batchnorm/ReadVariableOp9^model/layer_normalization_6/batchnorm/mul/ReadVariableOp5^model/layer_normalization_7/batchnorm/ReadVariableOp9^model/layer_normalization_7/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_2/BiasAdd/ReadVariableOp%model/conv1d_2/BiasAdd/ReadVariableOp2f
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_3/BiasAdd/ReadVariableOp%model/conv1d_3/BiasAdd/ReadVariableOp2f
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_4/BiasAdd/ReadVariableOp%model/conv1d_4/BiasAdd/ReadVariableOp2f
1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_5/BiasAdd/ReadVariableOp%model/conv1d_5/BiasAdd/ReadVariableOp2f
1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_6/BiasAdd/ReadVariableOp%model/conv1d_6/BiasAdd/ReadVariableOp2f
1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_7/BiasAdd/ReadVariableOp%model/conv1d_7/BiasAdd/ReadVariableOp2f
1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2N
%model/kf_attention/add/ReadVariableOp%model/kf_attention/add/ReadVariableOp2b
/model/kf_attention/einsum/Einsum/ReadVariableOp/model/kf_attention/einsum/Einsum/ReadVariableOp2f
1model/kf_attention/einsum_1/Einsum/ReadVariableOp1model/kf_attention/einsum_1/Einsum/ReadVariableOp2f
1model/kf_attention/einsum_4/Einsum/ReadVariableOp1model/kf_attention/einsum_4/Einsum/ReadVariableOp2R
'model/kf_attention_1/add/ReadVariableOp'model/kf_attention_1/add/ReadVariableOp2f
1model/kf_attention_1/einsum/Einsum/ReadVariableOp1model/kf_attention_1/einsum/Einsum/ReadVariableOp2j
3model/kf_attention_1/einsum_1/Einsum/ReadVariableOp3model/kf_attention_1/einsum_1/Einsum/ReadVariableOp2j
3model/kf_attention_1/einsum_4/Einsum/ReadVariableOp3model/kf_attention_1/einsum_4/Einsum/ReadVariableOp2R
'model/kf_attention_2/add/ReadVariableOp'model/kf_attention_2/add/ReadVariableOp2f
1model/kf_attention_2/einsum/Einsum/ReadVariableOp1model/kf_attention_2/einsum/Einsum/ReadVariableOp2j
3model/kf_attention_2/einsum_1/Einsum/ReadVariableOp3model/kf_attention_2/einsum_1/Einsum/ReadVariableOp2j
3model/kf_attention_2/einsum_4/Einsum/ReadVariableOp3model/kf_attention_2/einsum_4/Einsum/ReadVariableOp2R
'model/kf_attention_3/add/ReadVariableOp'model/kf_attention_3/add/ReadVariableOp2f
1model/kf_attention_3/einsum/Einsum/ReadVariableOp1model/kf_attention_3/einsum/Einsum/ReadVariableOp2j
3model/kf_attention_3/einsum_1/Einsum/ReadVariableOp3model/kf_attention_3/einsum_1/Einsum/ReadVariableOp2j
3model/kf_attention_3/einsum_4/Einsum/ReadVariableOp3model/kf_attention_3/einsum_4/Einsum/ReadVariableOp2h
2model/layer_normalization/batchnorm/ReadVariableOp2model/layer_normalization/batchnorm/ReadVariableOp2p
6model/layer_normalization/batchnorm/mul/ReadVariableOp6model/layer_normalization/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_1/batchnorm/ReadVariableOp4model/layer_normalization_1/batchnorm/ReadVariableOp2t
8model/layer_normalization_1/batchnorm/mul/ReadVariableOp8model/layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_2/batchnorm/ReadVariableOp4model/layer_normalization_2/batchnorm/ReadVariableOp2t
8model/layer_normalization_2/batchnorm/mul/ReadVariableOp8model/layer_normalization_2/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_3/batchnorm/ReadVariableOp4model/layer_normalization_3/batchnorm/ReadVariableOp2t
8model/layer_normalization_3/batchnorm/mul/ReadVariableOp8model/layer_normalization_3/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_4/batchnorm/ReadVariableOp4model/layer_normalization_4/batchnorm/ReadVariableOp2t
8model/layer_normalization_4/batchnorm/mul/ReadVariableOp8model/layer_normalization_4/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_5/batchnorm/ReadVariableOp4model/layer_normalization_5/batchnorm/ReadVariableOp2t
8model/layer_normalization_5/batchnorm/mul/ReadVariableOp8model/layer_normalization_5/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_6/batchnorm/ReadVariableOp4model/layer_normalization_6/batchnorm/ReadVariableOp2t
8model/layer_normalization_6/batchnorm/mul/ReadVariableOp8model/layer_normalization_6/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_7/batchnorm/ReadVariableOp4model/layer_normalization_7/batchnorm/ReadVariableOp2t
8model/layer_normalization_7/batchnorm/mul/ReadVariableOp8model/layer_normalization_7/batchnorm/mul/ReadVariableOp:T P
+
_output_shapes
:         

!
_user_specified_name	input_1
ш
Я
6__inference_layer_normalization_1_layer_call_fn_533393

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_529832s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
√
U
9__inference_global_average_pooling1d_layer_call_fn_534201

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_529686i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╚
У
D__inference_conv1d_2_layer_call_and_return_conditional_losses_530047

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┴
у
-__inference_kf_attention_layer_call_fn_533313

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529750s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_533415

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
в
╘
&__inference_model_layer_call_fn_531684
input_1
unknown:
	unknown_0: 
	unknown_1:А 
	unknown_2:А 
	unknown_3:А
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:!

unknown_13:А!

unknown_14:А!

unknown_15:А

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:!

unknown_25:А!

unknown_26:А!

unknown_27:А

unknown_28:

unknown_29:

unknown_30: 

unknown_31:

unknown_32: 

unknown_33:

unknown_34:

unknown_35:

unknown_36:!

unknown_37:А!

unknown_38:А!

unknown_39:А

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:	
А

unknown_48:	А

unknown_49:
А┘

unknown_50:	┘
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_531468p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         

!
_user_specified_name	input_1
Ё
У
D__inference_conv1d_1_layer_call_and_return_conditional_losses_533491

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_3_cond_true_5318631
kf_attention_3_cond_531867:А1
kf_attention_3_cond_531869:А1
kf_attention_3_cond_531871:А(
kf_attention_3_cond_531873:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_531867kf_attention_3_cond_531869kf_attention_3_cond_531871kf_attention_3_cond_531873*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530329п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

И
 
 kf_attention_2_cond_false_5318071
kf_attention_2_cond_531822:А1
kf_attention_2_cond_531824:А1
kf_attention_2_cond_531826:А(
kf_attention_2_cond_531828:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_531822kf_attention_2_cond_531824kf_attention_2_cond_531826kf_attention_2_cond_531828*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530170п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

┼
х
/__inference_kf_attention_2_layer_call_fn_533770

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530170s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_533925

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
а

Ї
A__inference_dense_layer_call_and_return_conditional_losses_534227

inputs1
matmul_readvariableop_resource:	
А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Б
c
*__inference_dropout_8_layer_call_fn_533920

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_530804s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_533619

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
У
D__inference_conv1d_3_layer_call_and_return_conditional_losses_533726

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┼
х
/__inference_kf_attention_3_layer_call_fn_534005

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530363s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_534120

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534040

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
п
F
*__inference_dropout_5_layer_call_fn_533680

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_530058d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
Я
6__inference_layer_normalization_4_layer_call_fn_533735

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_530104s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_533467

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533592

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530329

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_533702

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_530058

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
И
 
 kf_attention_1_cond_false_5319941
kf_attention_1_cond_532009:А1
kf_attention_1_cond_532011:А1
kf_attention_1_cond_532013:А(
kf_attention_1_cond_532015:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_532009kf_attention_1_cond_532011kf_attention_1_cond_532013kf_attention_1_cond_532015*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529977п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

Я
╙
&__inference_model_layer_call_fn_532503

inputs
unknown:
	unknown_0: 
	unknown_1:А 
	unknown_2:А 
	unknown_3:А
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:!

unknown_13:А!

unknown_14:А!

unknown_15:А

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:!

unknown_25:А!

unknown_26:А!

unknown_27:А

unknown_28:

unknown_29:

unknown_30: 

unknown_31:

unknown_32: 

unknown_33:

unknown_34:

unknown_35:

unknown_36:!

unknown_37:А!

unknown_38:А!

unknown_39:А

unknown_40:

unknown_41:

unknown_42: 

unknown_43:

unknown_44: 

unknown_45:

unknown_46:

unknown_47:	
А

unknown_48:	А

unknown_49:
А┘

unknown_50:	┘
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_531468p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Щк
Й
A__inference_model_layer_call_and_return_conditional_losses_530511

inputs(
layer_normalization_529719:(
layer_normalization_529721:0
kf_attention_cond_input_0:А0
kf_attention_cond_input_1:А0
kf_attention_cond_input_2:А'
kf_attention_cond_input_3:*
layer_normalization_1_529833:*
layer_normalization_1_529835:#
conv1d_529855:
conv1d_529857:%
conv1d_1_529883:
conv1d_1_529885:*
layer_normalization_2_529912:*
layer_normalization_2_529914:2
kf_attention_1_cond_input_0:А2
kf_attention_1_cond_input_1:А2
kf_attention_1_cond_input_2:А)
kf_attention_1_cond_input_3:*
layer_normalization_3_530026:*
layer_normalization_3_530028:%
conv1d_2_530048:
conv1d_2_530050:%
conv1d_3_530076:
conv1d_3_530078:*
layer_normalization_4_530105:*
layer_normalization_4_530107:2
kf_attention_2_cond_input_0:А2
kf_attention_2_cond_input_1:А2
kf_attention_2_cond_input_2:А)
kf_attention_2_cond_input_3:*
layer_normalization_5_530219:*
layer_normalization_5_530221:%
conv1d_4_530241:
conv1d_4_530243:%
conv1d_5_530269:
conv1d_5_530271:*
layer_normalization_6_530298:*
layer_normalization_6_530300:2
kf_attention_3_cond_input_0:А2
kf_attention_3_cond_input_1:А2
kf_attention_3_cond_input_2:А)
kf_attention_3_cond_input_3:*
layer_normalization_7_530412:*
layer_normalization_7_530414:%
conv1d_6_530434:
conv1d_6_530436:%
conv1d_7_530462:
conv1d_7_530464:
dense_530481:	
А
dense_530483:	А"
dense_1_530505:
А┘
dense_1_530507:	┘
identityИвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвkf_attention/condвkf_attention_1/condвkf_attention_2/condвkf_attention_3/condв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallв-layer_normalization_5/StatefulPartitionedCallв-layer_normalization_6/StatefulPartitionedCallв-layer_normalization_7/StatefulPartitionedCallа
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_529719layer_normalization_529721*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_529718}
kf_attention/cond/SqueezeSqueeze4layer_normalization/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:Т
kf_attention/condIf"kf_attention/cond/Squeeze:output:0kf_attention_cond_input_0kf_attention_cond_input_1kf_attention_cond_input_2kf_attention_cond_input_34layer_normalization/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*1
else_branch"R 
kf_attention_cond_false_529725**
output_shapes
:         
*0
then_branch!R
kf_attention_cond_true_529724x
kf_attention/cond/IdentityIdentitykf_attention/cond:output:0*
T0*+
_output_shapes
:         
█
dropout_1/PartitionedCallPartitionedCall#kf_attention/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_529807Е
tf.__operators__.add/AddV2AddV2"dropout_1/PartitionedCall:output:0inputs*
T0*+
_output_shapes
:         
└
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_529833layer_normalization_1_529835*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_529832Ь
conv1d/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_529855conv1d_529857*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_529854▀
dropout_2/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_529865Р
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_1_529883conv1d_1_529885*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_529882ж
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_529912layer_normalization_2_529914*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_529911Б
kf_attention_1/cond/SqueezeSqueeze6layer_normalization_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_1/condIf$kf_attention_1/cond/Squeeze:output:0kf_attention_1_cond_input_0kf_attention_1_cond_input_1kf_attention_1_cond_input_2kf_attention_1_cond_input_36layer_normalization_2/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_1_cond_false_529918**
output_shapes
:         
*2
then_branch#R!
kf_attention_1_cond_true_529917|
kf_attention_1/cond/IdentityIdentitykf_attention_1/cond:output:0*
T0*+
_output_shapes
:         
▌
dropout_4/PartitionedCallPartitionedCall%kf_attention_1/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_530000б
tf.__operators__.add_2/AddV2AddV2"dropout_4/PartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_3_530026layer_normalization_3_530028*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_530025д
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0conv1d_2_530048conv1d_2_530050*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_530047с
dropout_5/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_530058Р
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0conv1d_3_530076conv1d_3_530078*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_530075и
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_4_530105layer_normalization_4_530107*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_530104Б
kf_attention_2/cond/SqueezeSqueeze6layer_normalization_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_2/condIf$kf_attention_2/cond/Squeeze:output:0kf_attention_2_cond_input_0kf_attention_2_cond_input_1kf_attention_2_cond_input_2kf_attention_2_cond_input_36layer_normalization_4/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_2_cond_false_530111**
output_shapes
:         
*2
then_branch#R!
kf_attention_2_cond_true_530110|
kf_attention_2/cond/IdentityIdentitykf_attention_2/cond:output:0*
T0*+
_output_shapes
:         
▌
dropout_7/PartitionedCallPartitionedCall%kf_attention_2/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_530193б
tf.__operators__.add_4/AddV2AddV2"dropout_7/PartitionedCall:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_5_530219layer_normalization_5_530221*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_530218д
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0conv1d_4_530241conv1d_4_530243*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_530240с
dropout_8/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_530251Р
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv1d_5_530269conv1d_5_530271*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_530268и
tf.__operators__.add_5/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_6_530298layer_normalization_6_530300*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_530297Б
kf_attention_3/cond/SqueezeSqueeze6layer_normalization_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:д
kf_attention_3/condIf$kf_attention_3/cond/Squeeze:output:0kf_attention_3_cond_input_0kf_attention_3_cond_input_1kf_attention_3_cond_input_2kf_attention_3_cond_input_36layer_normalization_6/StatefulPartitionedCall:output:0*
Tcond0*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*+
_output_shapes
:         
*&
_read_only_resource_inputs
*3
else_branch$R"
 kf_attention_3_cond_false_530304**
output_shapes
:         
*2
then_branch#R!
kf_attention_3_cond_true_530303|
kf_attention_3/cond/IdentityIdentitykf_attention_3/cond:output:0*
T0*+
_output_shapes
:         
▀
dropout_10/PartitionedCallPartitionedCall%kf_attention_3/cond/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_530386в
tf.__operators__.add_6/AddV2AddV2#dropout_10/PartitionedCall:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
┬
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0layer_normalization_7_530412layer_normalization_7_530414*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_530411д
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0conv1d_6_530434conv1d_6_530436*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_530433у
dropout_11/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_530444С
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0conv1d_7_530462conv1d_7_530464*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_530461и
tf.__operators__.add_7/AddV2AddV2)conv1d_7/StatefulPartitionedCall:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
Є
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_7/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_529686Р
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_530481dense_530483*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_530480▌
dropout_12/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_530491К
dense_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_1_530505dense_1_530507*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┘*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_530504x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┘Є
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^kf_attention/cond^kf_attention_1/cond^kf_attention_2/cond^kf_attention_3/cond,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2&
kf_attention/condkf_attention/cond2*
kf_attention_1/condkf_attention_1/cond2*
kf_attention_2/condkf_attention_2/cond2*
kf_attention_3/condkf_attention_3/cond2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Л
╞
H__inference_kf_attention_layer_call_and_return_conditional_losses_533335

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
и

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_534172

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530363

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Г
d
+__inference_dropout_11_layer_call_fn_534155

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_530691s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
А█
▌.
A__inference_model_layer_call_and_return_conditional_losses_532848

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:I
2kf_attention_einsum_einsum_readvariableop_resource:АK
4kf_attention_einsum_1_einsum_readvariableop_resource:АK
4kf_attention_einsum_4_einsum_readvariableop_resource:А6
(kf_attention_add_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:K
4kf_attention_1_einsum_einsum_readvariableop_resource:АM
6kf_attention_1_einsum_1_einsum_readvariableop_resource:АM
6kf_attention_1_einsum_4_einsum_readvariableop_resource:А8
*kf_attention_1_add_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_3_biasadd_readvariableop_resource:I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:E
7layer_normalization_4_batchnorm_readvariableop_resource:K
4kf_attention_2_einsum_einsum_readvariableop_resource:АM
6kf_attention_2_einsum_1_einsum_readvariableop_resource:АM
6kf_attention_2_einsum_4_einsum_readvariableop_resource:А8
*kf_attention_2_add_readvariableop_resource:I
;layer_normalization_5_batchnorm_mul_readvariableop_resource:E
7layer_normalization_5_batchnorm_readvariableop_resource:J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_4_biasadd_readvariableop_resource:J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_5_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:K
4kf_attention_3_einsum_einsum_readvariableop_resource:АM
6kf_attention_3_einsum_1_einsum_readvariableop_resource:АM
6kf_attention_3_einsum_4_einsum_readvariableop_resource:А8
*kf_attention_3_add_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:J
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_6_biasadd_readvariableop_resource:J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_7_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	
А4
%dense_biasadd_readvariableop_resource:	А:
&dense_1_matmul_readvariableop_resource:
А┘6
'dense_1_biasadd_readvariableop_resource:	┘
identityИвconv1d/BiasAdd/ReadVariableOpв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_1/BiasAdd/ReadVariableOpв+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_2/BiasAdd/ReadVariableOpв+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_3/BiasAdd/ReadVariableOpв+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_4/BiasAdd/ReadVariableOpв+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_5/BiasAdd/ReadVariableOpв+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_6/BiasAdd/ReadVariableOpв+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_7/BiasAdd/ReadVariableOpв+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвkf_attention/add/ReadVariableOpв)kf_attention/einsum/Einsum/ReadVariableOpв+kf_attention/einsum_1/Einsum/ReadVariableOpв+kf_attention/einsum_4/Einsum/ReadVariableOpв!kf_attention_1/add/ReadVariableOpв+kf_attention_1/einsum/Einsum/ReadVariableOpв-kf_attention_1/einsum_1/Einsum/ReadVariableOpв-kf_attention_1/einsum_4/Einsum/ReadVariableOpв!kf_attention_2/add/ReadVariableOpв+kf_attention_2/einsum/Einsum/ReadVariableOpв-kf_attention_2/einsum_1/Einsum/ReadVariableOpв-kf_attention_2/einsum_4/Einsum/ReadVariableOpв!kf_attention_3/add/ReadVariableOpв+kf_attention_3/einsum/Einsum/ReadVariableOpв-kf_attention_3/einsum_1/Einsum/ReadVariableOpв-kf_attention_3/einsum_4/Einsum/ReadVariableOpв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpв.layer_normalization_2/batchnorm/ReadVariableOpв2layer_normalization_2/batchnorm/mul/ReadVariableOpв.layer_normalization_3/batchnorm/ReadVariableOpв2layer_normalization_3/batchnorm/mul/ReadVariableOpв.layer_normalization_4/batchnorm/ReadVariableOpв2layer_normalization_4/batchnorm/mul/ReadVariableOpв.layer_normalization_5/batchnorm/ReadVariableOpв2layer_normalization_5/batchnorm/mul/ReadVariableOpв.layer_normalization_6/batchnorm/ReadVariableOpв2layer_normalization_6/batchnorm/mul/ReadVariableOpв.layer_normalization_7/batchnorm/ReadVariableOpв2layer_normalization_7/batchnorm/mul/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Щ
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         
│
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ч
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╜
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Й
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         
ж
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┴
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
П
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
▓
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
Ю
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╜
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
▓
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
б
)kf_attention/einsum/Einsum/ReadVariableOpReadVariableOp2kf_attention_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0х
kf_attention/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:01kf_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOе
+kf_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp4kf_attention_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0щ
kf_attention/einsum_1/EinsumEinsum'layer_normalization/batchnorm/add_1:z:03kf_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOW
kf_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АCW
kf_attention/SqrtSqrtkf_attention/Const:output:0*
T0*
_output_shapes
: Ц
kf_attention/truedivRealDiv#kf_attention/einsum/Einsum:output:0kf_attention/Sqrt:y:0*
T0*0
_output_shapes
:         
А╧
kf_attention/einsum_2/EinsumEinsumkf_attention/truediv:z:0%kf_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMА
kf_attention/SoftmaxSoftmax%kf_attention/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

╓
kf_attention/einsum_3/EinsumEinsumkf_attention/Softmax:softmax:0%kf_attention/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIе
+kf_attention/einsum_4/Einsum/ReadVariableOpReadVariableOp4kf_attention_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0т
kf_attention/einsum_4/EinsumEinsum%kf_attention/einsum_3/Einsum:output:03kf_attention/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOД
kf_attention/add/ReadVariableOpReadVariableOp(kf_attention_add_readvariableop_resource*
_output_shapes
:*
dtype0Я
kf_attention/addAddV2%kf_attention/einsum_4/Einsum:output:0'kf_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
j
dropout_1/IdentityIdentitykf_attention/add:z:0*
T0*+
_output_shapes
:         
~
tf.__operators__.add/AddV2AddV2dropout_1/Identity:output:0inputs*
T0*+
_output_shapes
:         
~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╨
"layer_normalization_1/moments/meanMeantf.__operators__.add/AddV2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         
╧
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetf.__operators__.add/AddV2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
л
%layer_normalization_1/batchnorm/mul_1Multf.__operators__.add/AddV2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ▓
conv1d/Conv1D/ExpandDims
ExpandDims)layer_normalization_1/batchnorm/add_1:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┬
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        А
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ц
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:         
o
dropout_2/IdentityIdentityconv1d/Relu:activations:0*
T0*+
_output_shapes
:         
i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        и
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_2/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ц
tf.__operators__.add_1/AddV2AddV2conv1d_1/BiasAdd:output:0tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_2/moments/meanMean tf.__operators__.add_1/AddV2:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_2/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_1/AddV2:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_2/batchnorm/mul_1Mul tf.__operators__.add_1/AddV2:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
е
+kf_attention_1/einsum/Einsum/ReadVariableOpReadVariableOp4kf_attention_1_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ы
kf_attention_1/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:03kf_attention_1/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOй
-kf_attention_1/einsum_1/Einsum/ReadVariableOpReadVariableOp6kf_attention_1_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0я
kf_attention_1/einsum_1/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:05kf_attention_1/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOY
kf_attention_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC[
kf_attention_1/SqrtSqrtkf_attention_1/Const:output:0*
T0*
_output_shapes
: Ь
kf_attention_1/truedivRealDiv%kf_attention_1/einsum/Einsum:output:0kf_attention_1/Sqrt:y:0*
T0*0
_output_shapes
:         
А╒
kf_attention_1/einsum_2/EinsumEinsumkf_attention_1/truediv:z:0'kf_attention_1/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMД
kf_attention_1/SoftmaxSoftmax'kf_attention_1/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

▄
kf_attention_1/einsum_3/EinsumEinsum kf_attention_1/Softmax:softmax:0'kf_attention_1/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIй
-kf_attention_1/einsum_4/Einsum/ReadVariableOpReadVariableOp6kf_attention_1_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ш
kf_attention_1/einsum_4/EinsumEinsum'kf_attention_1/einsum_3/Einsum:output:05kf_attention_1/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOИ
!kf_attention_1/add/ReadVariableOpReadVariableOp*kf_attention_1_add_readvariableop_resource*
_output_shapes
:*
dtype0е
kf_attention_1/addAddV2'kf_attention_1/einsum_4/Einsum:output:0)kf_attention_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
l
dropout_4/IdentityIdentitykf_attention_1/add:z:0*
T0*+
_output_shapes
:         
Ъ
tf.__operators__.add_2/AddV2AddV2dropout_4/Identity:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_3/moments/meanMean tf.__operators__.add_2/AddV2:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_2/AddV2:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_3/batchnorm/mul_1Mul tf.__operators__.add_2/AddV2:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╢
conv1d_2/Conv1D/ExpandDims
ExpandDims)layer_normalization_3/batchnorm/add_1:z:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:         
q
dropout_5/IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:         
i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        и
conv1d_3/Conv1D/ExpandDims
ExpandDimsdropout_5/Identity:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ш
tf.__operators__.add_3/AddV2AddV2conv1d_3/BiasAdd:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_4/moments/meanMean tf.__operators__.add_3/AddV2:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_4/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_3/AddV2:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_4/batchnorm/mul_1Mul tf.__operators__.add_3/AddV2:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
е
+kf_attention_2/einsum/Einsum/ReadVariableOpReadVariableOp4kf_attention_2_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ы
kf_attention_2/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:03kf_attention_2/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOй
-kf_attention_2/einsum_1/Einsum/ReadVariableOpReadVariableOp6kf_attention_2_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0я
kf_attention_2/einsum_1/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:05kf_attention_2/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOY
kf_attention_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC[
kf_attention_2/SqrtSqrtkf_attention_2/Const:output:0*
T0*
_output_shapes
: Ь
kf_attention_2/truedivRealDiv%kf_attention_2/einsum/Einsum:output:0kf_attention_2/Sqrt:y:0*
T0*0
_output_shapes
:         
А╒
kf_attention_2/einsum_2/EinsumEinsumkf_attention_2/truediv:z:0'kf_attention_2/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMД
kf_attention_2/SoftmaxSoftmax'kf_attention_2/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

▄
kf_attention_2/einsum_3/EinsumEinsum kf_attention_2/Softmax:softmax:0'kf_attention_2/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIй
-kf_attention_2/einsum_4/Einsum/ReadVariableOpReadVariableOp6kf_attention_2_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ш
kf_attention_2/einsum_4/EinsumEinsum'kf_attention_2/einsum_3/Einsum:output:05kf_attention_2/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOИ
!kf_attention_2/add/ReadVariableOpReadVariableOp*kf_attention_2_add_readvariableop_resource*
_output_shapes
:*
dtype0е
kf_attention_2/addAddV2'kf_attention_2/einsum_4/Einsum:output:0)kf_attention_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
l
dropout_7/IdentityIdentitykf_attention_2/add:z:0*
T0*+
_output_shapes
:         
Ъ
tf.__operators__.add_4/AddV2AddV2dropout_7/Identity:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_5/moments/meanMean tf.__operators__.add_4/AddV2:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_4/AddV2:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_5/batchnorm/mul_1Mul tf.__operators__.add_4/AddV2:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
i
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╢
conv1d_4/Conv1D/ExpandDims
ExpandDims)layer_normalization_5/batchnorm/add_1:z:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
f
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:         
q
dropout_8/IdentityIdentityconv1d_4/Relu:activations:0*
T0*+
_output_shapes
:         
i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        и
conv1d_5/Conv1D/ExpandDims
ExpandDimsdropout_8/Identity:output:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ш
tf.__operators__.add_5/AddV2AddV2conv1d_5/BiasAdd:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_6/moments/meanMean tf.__operators__.add_5/AddV2:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_6/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_5/AddV2:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_6/batchnorm/mul_1Mul tf.__operators__.add_5/AddV2:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
е
+kf_attention_3/einsum/Einsum/ReadVariableOpReadVariableOp4kf_attention_3_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ы
kf_attention_3/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:03kf_attention_3/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOй
-kf_attention_3/einsum_1/Einsum/ReadVariableOpReadVariableOp6kf_attention_3_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0я
kf_attention_3/einsum_1/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:05kf_attention_3/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOY
kf_attention_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC[
kf_attention_3/SqrtSqrtkf_attention_3/Const:output:0*
T0*
_output_shapes
: Ь
kf_attention_3/truedivRealDiv%kf_attention_3/einsum/Einsum:output:0kf_attention_3/Sqrt:y:0*
T0*0
_output_shapes
:         
А╒
kf_attention_3/einsum_2/EinsumEinsumkf_attention_3/truediv:z:0'kf_attention_3/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMД
kf_attention_3/SoftmaxSoftmax'kf_attention_3/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

▄
kf_attention_3/einsum_3/EinsumEinsum kf_attention_3/Softmax:softmax:0'kf_attention_3/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIй
-kf_attention_3/einsum_4/Einsum/ReadVariableOpReadVariableOp6kf_attention_3_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ш
kf_attention_3/einsum_4/EinsumEinsum'kf_attention_3/einsum_3/Einsum:output:05kf_attention_3/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOИ
!kf_attention_3/add/ReadVariableOpReadVariableOp*kf_attention_3_add_readvariableop_resource*
_output_shapes
:*
dtype0е
kf_attention_3/addAddV2'kf_attention_3/einsum_4/Einsum:output:0)kf_attention_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
m
dropout_10/IdentityIdentitykf_attention_3/add:z:0*
T0*+
_output_shapes
:         
Ы
tf.__operators__.add_6/AddV2AddV2dropout_10/Identity:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_7/moments/meanMean tf.__operators__.add_6/AddV2:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_6/AddV2:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_7/batchnorm/mul_1Mul tf.__operators__.add_6/AddV2:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
i
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╢
conv1d_6/Conv1D/ExpandDims
ExpandDims)layer_normalization_7/batchnorm/add_1:z:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_6/BiasAddBiasAdd conv1d_6/Conv1D/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
f
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:         
r
dropout_11/IdentityIdentityconv1d_6/Relu:activations:0*
T0*+
_output_shapes
:         
i
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        й
conv1d_7/Conv1D/ExpandDims
ExpandDimsdropout_11/Identity:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_7/BiasAddBiasAdd conv1d_7/Conv1D/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ш
tf.__operators__.add_7/AddV2AddV2conv1d_7/BiasAdd:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :│
global_average_pooling1d/MeanMean tf.__operators__.add_7/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         
Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	
А*
dtype0Ц
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         Аl
dropout_12/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:         АЖ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
А┘*
dtype0Р
dense_1/MatMulMatMuldropout_12/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘Г
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:┘*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘g
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ┘i
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ┘м
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^kf_attention/add/ReadVariableOp*^kf_attention/einsum/Einsum/ReadVariableOp,^kf_attention/einsum_1/Einsum/ReadVariableOp,^kf_attention/einsum_4/Einsum/ReadVariableOp"^kf_attention_1/add/ReadVariableOp,^kf_attention_1/einsum/Einsum/ReadVariableOp.^kf_attention_1/einsum_1/Einsum/ReadVariableOp.^kf_attention_1/einsum_4/Einsum/ReadVariableOp"^kf_attention_2/add/ReadVariableOp,^kf_attention_2/einsum/Einsum/ReadVariableOp.^kf_attention_2/einsum_1/Einsum/ReadVariableOp.^kf_attention_2/einsum_4/Einsum/ReadVariableOp"^kf_attention_3/add/ReadVariableOp,^kf_attention_3/einsum/Einsum/ReadVariableOp.^kf_attention_3/einsum_1/Einsum/ReadVariableOp.^kf_attention_3/einsum_4/Einsum/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
kf_attention/add/ReadVariableOpkf_attention/add/ReadVariableOp2V
)kf_attention/einsum/Einsum/ReadVariableOp)kf_attention/einsum/Einsum/ReadVariableOp2Z
+kf_attention/einsum_1/Einsum/ReadVariableOp+kf_attention/einsum_1/Einsum/ReadVariableOp2Z
+kf_attention/einsum_4/Einsum/ReadVariableOp+kf_attention/einsum_4/Einsum/ReadVariableOp2F
!kf_attention_1/add/ReadVariableOp!kf_attention_1/add/ReadVariableOp2Z
+kf_attention_1/einsum/Einsum/ReadVariableOp+kf_attention_1/einsum/Einsum/ReadVariableOp2^
-kf_attention_1/einsum_1/Einsum/ReadVariableOp-kf_attention_1/einsum_1/Einsum/ReadVariableOp2^
-kf_attention_1/einsum_4/Einsum/ReadVariableOp-kf_attention_1/einsum_4/Einsum/ReadVariableOp2F
!kf_attention_2/add/ReadVariableOp!kf_attention_2/add/ReadVariableOp2Z
+kf_attention_2/einsum/Einsum/ReadVariableOp+kf_attention_2/einsum/Einsum/ReadVariableOp2^
-kf_attention_2/einsum_1/Einsum/ReadVariableOp-kf_attention_2/einsum_1/Einsum/ReadVariableOp2^
-kf_attention_2/einsum_4/Einsum/ReadVariableOp-kf_attention_2/einsum_4/Einsum/ReadVariableOp2F
!kf_attention_3/add/ReadVariableOp!kf_attention_3/add/ReadVariableOp2Z
+kf_attention_3/einsum/Einsum/ReadVariableOp+kf_attention_3/einsum/Einsum/ReadVariableOp2^
-kf_attention_3/einsum_1/Einsum/ReadVariableOp-kf_attention_3/einsum_1/Einsum/ReadVariableOp2^
-kf_attention_3/einsum_4/Einsum/ReadVariableOp-kf_attention_3/einsum_4/Einsum/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
а

Ї
A__inference_dense_layer_call_and_return_conditional_losses_530480

inputs1
matmul_readvariableop_resource:	
А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
М
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_534207

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╚
У
D__inference_conv1d_4_layer_call_and_return_conditional_losses_530240

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Н
╚
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530136

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ж┤
уo
"__inference__traced_restore_535279
file_prefix8
*assignvariableop_layer_normalization_gamma:9
+assignvariableop_1_layer_normalization_beta:C
,assignvariableop_2_kf_attention_query_kernel:АC
,assignvariableop_3_kf_attention_value_kernel:АH
1assignvariableop_4_kf_attention_projection_kernel:А=
/assignvariableop_5_kf_attention_projection_bias:<
.assignvariableop_6_layer_normalization_1_gamma:;
-assignvariableop_7_layer_normalization_1_beta:6
 assignvariableop_8_conv1d_kernel:,
assignvariableop_9_conv1d_bias:9
#assignvariableop_10_conv1d_1_kernel:/
!assignvariableop_11_conv1d_1_bias:=
/assignvariableop_12_layer_normalization_2_gamma:<
.assignvariableop_13_layer_normalization_2_beta:F
/assignvariableop_14_kf_attention_1_query_kernel:АF
/assignvariableop_15_kf_attention_1_value_kernel:АK
4assignvariableop_16_kf_attention_1_projection_kernel:А@
2assignvariableop_17_kf_attention_1_projection_bias:=
/assignvariableop_18_layer_normalization_3_gamma:<
.assignvariableop_19_layer_normalization_3_beta:9
#assignvariableop_20_conv1d_2_kernel:/
!assignvariableop_21_conv1d_2_bias:9
#assignvariableop_22_conv1d_3_kernel:/
!assignvariableop_23_conv1d_3_bias:=
/assignvariableop_24_layer_normalization_4_gamma:<
.assignvariableop_25_layer_normalization_4_beta:F
/assignvariableop_26_kf_attention_2_query_kernel:АF
/assignvariableop_27_kf_attention_2_value_kernel:АK
4assignvariableop_28_kf_attention_2_projection_kernel:А@
2assignvariableop_29_kf_attention_2_projection_bias:=
/assignvariableop_30_layer_normalization_5_gamma:<
.assignvariableop_31_layer_normalization_5_beta:9
#assignvariableop_32_conv1d_4_kernel:/
!assignvariableop_33_conv1d_4_bias:9
#assignvariableop_34_conv1d_5_kernel:/
!assignvariableop_35_conv1d_5_bias:=
/assignvariableop_36_layer_normalization_6_gamma:<
.assignvariableop_37_layer_normalization_6_beta:F
/assignvariableop_38_kf_attention_3_query_kernel:АF
/assignvariableop_39_kf_attention_3_value_kernel:АK
4assignvariableop_40_kf_attention_3_projection_kernel:А@
2assignvariableop_41_kf_attention_3_projection_bias:=
/assignvariableop_42_layer_normalization_7_gamma:<
.assignvariableop_43_layer_normalization_7_beta:9
#assignvariableop_44_conv1d_6_kernel:/
!assignvariableop_45_conv1d_6_bias:9
#assignvariableop_46_conv1d_7_kernel:/
!assignvariableop_47_conv1d_7_bias:3
 assignvariableop_48_dense_kernel:	
А-
assignvariableop_49_dense_bias:	А6
"assignvariableop_50_dense_1_kernel:
А┘/
 assignvariableop_51_dense_1_bias:	┘'
assignvariableop_52_iteration:	 +
!assignvariableop_53_learning_rate: B
4assignvariableop_54_adam_m_layer_normalization_gamma:B
4assignvariableop_55_adam_v_layer_normalization_gamma:A
3assignvariableop_56_adam_m_layer_normalization_beta:A
3assignvariableop_57_adam_v_layer_normalization_beta:K
4assignvariableop_58_adam_m_kf_attention_query_kernel:АK
4assignvariableop_59_adam_v_kf_attention_query_kernel:АK
4assignvariableop_60_adam_m_kf_attention_value_kernel:АK
4assignvariableop_61_adam_v_kf_attention_value_kernel:АP
9assignvariableop_62_adam_m_kf_attention_projection_kernel:АP
9assignvariableop_63_adam_v_kf_attention_projection_kernel:АE
7assignvariableop_64_adam_m_kf_attention_projection_bias:E
7assignvariableop_65_adam_v_kf_attention_projection_bias:D
6assignvariableop_66_adam_m_layer_normalization_1_gamma:D
6assignvariableop_67_adam_v_layer_normalization_1_gamma:C
5assignvariableop_68_adam_m_layer_normalization_1_beta:C
5assignvariableop_69_adam_v_layer_normalization_1_beta:>
(assignvariableop_70_adam_m_conv1d_kernel:>
(assignvariableop_71_adam_v_conv1d_kernel:4
&assignvariableop_72_adam_m_conv1d_bias:4
&assignvariableop_73_adam_v_conv1d_bias:@
*assignvariableop_74_adam_m_conv1d_1_kernel:@
*assignvariableop_75_adam_v_conv1d_1_kernel:6
(assignvariableop_76_adam_m_conv1d_1_bias:6
(assignvariableop_77_adam_v_conv1d_1_bias:D
6assignvariableop_78_adam_m_layer_normalization_2_gamma:D
6assignvariableop_79_adam_v_layer_normalization_2_gamma:C
5assignvariableop_80_adam_m_layer_normalization_2_beta:C
5assignvariableop_81_adam_v_layer_normalization_2_beta:M
6assignvariableop_82_adam_m_kf_attention_1_query_kernel:АM
6assignvariableop_83_adam_v_kf_attention_1_query_kernel:АM
6assignvariableop_84_adam_m_kf_attention_1_value_kernel:АM
6assignvariableop_85_adam_v_kf_attention_1_value_kernel:АR
;assignvariableop_86_adam_m_kf_attention_1_projection_kernel:АR
;assignvariableop_87_adam_v_kf_attention_1_projection_kernel:АG
9assignvariableop_88_adam_m_kf_attention_1_projection_bias:G
9assignvariableop_89_adam_v_kf_attention_1_projection_bias:D
6assignvariableop_90_adam_m_layer_normalization_3_gamma:D
6assignvariableop_91_adam_v_layer_normalization_3_gamma:C
5assignvariableop_92_adam_m_layer_normalization_3_beta:C
5assignvariableop_93_adam_v_layer_normalization_3_beta:@
*assignvariableop_94_adam_m_conv1d_2_kernel:@
*assignvariableop_95_adam_v_conv1d_2_kernel:6
(assignvariableop_96_adam_m_conv1d_2_bias:6
(assignvariableop_97_adam_v_conv1d_2_bias:@
*assignvariableop_98_adam_m_conv1d_3_kernel:@
*assignvariableop_99_adam_v_conv1d_3_kernel:7
)assignvariableop_100_adam_m_conv1d_3_bias:7
)assignvariableop_101_adam_v_conv1d_3_bias:E
7assignvariableop_102_adam_m_layer_normalization_4_gamma:E
7assignvariableop_103_adam_v_layer_normalization_4_gamma:D
6assignvariableop_104_adam_m_layer_normalization_4_beta:D
6assignvariableop_105_adam_v_layer_normalization_4_beta:N
7assignvariableop_106_adam_m_kf_attention_2_query_kernel:АN
7assignvariableop_107_adam_v_kf_attention_2_query_kernel:АN
7assignvariableop_108_adam_m_kf_attention_2_value_kernel:АN
7assignvariableop_109_adam_v_kf_attention_2_value_kernel:АS
<assignvariableop_110_adam_m_kf_attention_2_projection_kernel:АS
<assignvariableop_111_adam_v_kf_attention_2_projection_kernel:АH
:assignvariableop_112_adam_m_kf_attention_2_projection_bias:H
:assignvariableop_113_adam_v_kf_attention_2_projection_bias:E
7assignvariableop_114_adam_m_layer_normalization_5_gamma:E
7assignvariableop_115_adam_v_layer_normalization_5_gamma:D
6assignvariableop_116_adam_m_layer_normalization_5_beta:D
6assignvariableop_117_adam_v_layer_normalization_5_beta:A
+assignvariableop_118_adam_m_conv1d_4_kernel:A
+assignvariableop_119_adam_v_conv1d_4_kernel:7
)assignvariableop_120_adam_m_conv1d_4_bias:7
)assignvariableop_121_adam_v_conv1d_4_bias:A
+assignvariableop_122_adam_m_conv1d_5_kernel:A
+assignvariableop_123_adam_v_conv1d_5_kernel:7
)assignvariableop_124_adam_m_conv1d_5_bias:7
)assignvariableop_125_adam_v_conv1d_5_bias:E
7assignvariableop_126_adam_m_layer_normalization_6_gamma:E
7assignvariableop_127_adam_v_layer_normalization_6_gamma:D
6assignvariableop_128_adam_m_layer_normalization_6_beta:D
6assignvariableop_129_adam_v_layer_normalization_6_beta:N
7assignvariableop_130_adam_m_kf_attention_3_query_kernel:АN
7assignvariableop_131_adam_v_kf_attention_3_query_kernel:АN
7assignvariableop_132_adam_m_kf_attention_3_value_kernel:АN
7assignvariableop_133_adam_v_kf_attention_3_value_kernel:АS
<assignvariableop_134_adam_m_kf_attention_3_projection_kernel:АS
<assignvariableop_135_adam_v_kf_attention_3_projection_kernel:АH
:assignvariableop_136_adam_m_kf_attention_3_projection_bias:H
:assignvariableop_137_adam_v_kf_attention_3_projection_bias:E
7assignvariableop_138_adam_m_layer_normalization_7_gamma:E
7assignvariableop_139_adam_v_layer_normalization_7_gamma:D
6assignvariableop_140_adam_m_layer_normalization_7_beta:D
6assignvariableop_141_adam_v_layer_normalization_7_beta:A
+assignvariableop_142_adam_m_conv1d_6_kernel:A
+assignvariableop_143_adam_v_conv1d_6_kernel:7
)assignvariableop_144_adam_m_conv1d_6_bias:7
)assignvariableop_145_adam_v_conv1d_6_bias:A
+assignvariableop_146_adam_m_conv1d_7_kernel:A
+assignvariableop_147_adam_v_conv1d_7_kernel:7
)assignvariableop_148_adam_m_conv1d_7_bias:7
)assignvariableop_149_adam_v_conv1d_7_bias:;
(assignvariableop_150_adam_m_dense_kernel:	
А;
(assignvariableop_151_adam_v_dense_kernel:	
А5
&assignvariableop_152_adam_m_dense_bias:	А5
&assignvariableop_153_adam_v_dense_bias:	А>
*assignvariableop_154_adam_m_dense_1_kernel:
А┘>
*assignvariableop_155_adam_v_dense_1_kernel:
А┘7
(assignvariableop_156_adam_m_dense_1_bias:	┘7
(assignvariableop_157_adam_v_dense_1_bias:	┘&
assignvariableop_158_total_1: &
assignvariableop_159_count_1: $
assignvariableop_160_total: $
assignvariableop_161_count: 
identity_163ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_110вAssignVariableOp_111вAssignVariableOp_112вAssignVariableOp_113вAssignVariableOp_114вAssignVariableOp_115вAssignVariableOp_116вAssignVariableOp_117вAssignVariableOp_118вAssignVariableOp_119вAssignVariableOp_12вAssignVariableOp_120вAssignVariableOp_121вAssignVariableOp_122вAssignVariableOp_123вAssignVariableOp_124вAssignVariableOp_125вAssignVariableOp_126вAssignVariableOp_127вAssignVariableOp_128вAssignVariableOp_129вAssignVariableOp_13вAssignVariableOp_130вAssignVariableOp_131вAssignVariableOp_132вAssignVariableOp_133вAssignVariableOp_134вAssignVariableOp_135вAssignVariableOp_136вAssignVariableOp_137вAssignVariableOp_138вAssignVariableOp_139вAssignVariableOp_14вAssignVariableOp_140вAssignVariableOp_141вAssignVariableOp_142вAssignVariableOp_143вAssignVariableOp_144вAssignVariableOp_145вAssignVariableOp_146вAssignVariableOp_147вAssignVariableOp_148вAssignVariableOp_149вAssignVariableOp_15вAssignVariableOp_150вAssignVariableOp_151вAssignVariableOp_152вAssignVariableOp_153вAssignVariableOp_154вAssignVariableOp_155вAssignVariableOp_156вAssignVariableOp_157вAssignVariableOp_158вAssignVariableOp_159вAssignVariableOp_16вAssignVariableOp_160вAssignVariableOp_161вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99║E
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:г*
dtype0*▀D
value╒DB╥DгB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-6/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-6/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-6/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-11/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-11/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-11/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-16/query_kernel/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-16/value_kernel/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-16/projection_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/projection_bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/93/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/94/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/95/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/96/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/97/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/98/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/99/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/100/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/101/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/102/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/103/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_variables/104/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╗
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:г*
dtype0*▄
value╥B╧гB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╘
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*в
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*┤
dtypesй
ж2г	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_2AssignVariableOp,assignvariableop_2_kf_attention_query_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_3AssignVariableOp,assignvariableop_3_kf_attention_value_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_4AssignVariableOp1assignvariableop_4_kf_attention_projection_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_5AssignVariableOp/assignvariableop_5_kf_attention_projection_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_1_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv1d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv1d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv1d_1_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv1d_1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_12AssignVariableOp/assignvariableop_12_layer_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_13AssignVariableOp.assignvariableop_13_layer_normalization_2_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_14AssignVariableOp/assignvariableop_14_kf_attention_1_query_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_15AssignVariableOp/assignvariableop_15_kf_attention_1_value_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_16AssignVariableOp4assignvariableop_16_kf_attention_1_projection_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_17AssignVariableOp2assignvariableop_17_kf_attention_1_projection_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_18AssignVariableOp/assignvariableop_18_layer_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_19AssignVariableOp.assignvariableop_19_layer_normalization_3_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv1d_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv1d_2_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv1d_3_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv1d_3_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_24AssignVariableOp/assignvariableop_24_layer_normalization_4_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_25AssignVariableOp.assignvariableop_25_layer_normalization_4_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_26AssignVariableOp/assignvariableop_26_kf_attention_2_query_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_27AssignVariableOp/assignvariableop_27_kf_attention_2_value_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_28AssignVariableOp4assignvariableop_28_kf_attention_2_projection_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_29AssignVariableOp2assignvariableop_29_kf_attention_2_projection_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_30AssignVariableOp/assignvariableop_30_layer_normalization_5_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_31AssignVariableOp.assignvariableop_31_layer_normalization_5_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv1d_4_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_33AssignVariableOp!assignvariableop_33_conv1d_4_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_34AssignVariableOp#assignvariableop_34_conv1d_5_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_35AssignVariableOp!assignvariableop_35_conv1d_5_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_36AssignVariableOp/assignvariableop_36_layer_normalization_6_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_37AssignVariableOp.assignvariableop_37_layer_normalization_6_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_38AssignVariableOp/assignvariableop_38_kf_attention_3_query_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_39AssignVariableOp/assignvariableop_39_kf_attention_3_value_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_40AssignVariableOp4assignvariableop_40_kf_attention_3_projection_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_41AssignVariableOp2assignvariableop_41_kf_attention_3_projection_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_42AssignVariableOp/assignvariableop_42_layer_normalization_7_gammaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_43AssignVariableOp.assignvariableop_43_layer_normalization_7_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv1d_6_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_45AssignVariableOp!assignvariableop_45_conv1d_6_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_46AssignVariableOp#assignvariableop_46_conv1d_7_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_47AssignVariableOp!assignvariableop_47_conv1d_7_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_48AssignVariableOp assignvariableop_48_dense_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_49AssignVariableOpassignvariableop_49_dense_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_50AssignVariableOp"assignvariableop_50_dense_1_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_51AssignVariableOp assignvariableop_51_dense_1_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_52AssignVariableOpassignvariableop_52_iterationIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_53AssignVariableOp!assignvariableop_53_learning_rateIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_54AssignVariableOp4assignvariableop_54_adam_m_layer_normalization_gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adam_v_layer_normalization_gammaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_m_layer_normalization_betaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_57AssignVariableOp3assignvariableop_57_adam_v_layer_normalization_betaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_58AssignVariableOp4assignvariableop_58_adam_m_kf_attention_query_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_v_kf_attention_query_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_60AssignVariableOp4assignvariableop_60_adam_m_kf_attention_value_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_v_kf_attention_value_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_62AssignVariableOp9assignvariableop_62_adam_m_kf_attention_projection_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_63AssignVariableOp9assignvariableop_63_adam_v_kf_attention_projection_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_m_kf_attention_projection_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_v_kf_attention_projection_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_m_layer_normalization_1_gammaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_v_layer_normalization_1_gammaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_m_layer_normalization_1_betaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_v_layer_normalization_1_betaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_m_conv1d_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_v_conv1d_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_72AssignVariableOp&assignvariableop_72_adam_m_conv1d_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_73AssignVariableOp&assignvariableop_73_adam_v_conv1d_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_m_conv1d_1_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_v_conv1d_1_kernelIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_m_conv1d_1_biasIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_v_conv1d_1_biasIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_m_layer_normalization_2_gammaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_v_layer_normalization_2_gammaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adam_m_layer_normalization_2_betaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_81AssignVariableOp5assignvariableop_81_adam_v_layer_normalization_2_betaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_m_kf_attention_1_query_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_83AssignVariableOp6assignvariableop_83_adam_v_kf_attention_1_query_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_m_kf_attention_1_value_kernelIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_v_kf_attention_1_value_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_86AssignVariableOp;assignvariableop_86_adam_m_kf_attention_1_projection_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_87AssignVariableOp;assignvariableop_87_adam_v_kf_attention_1_projection_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_88AssignVariableOp9assignvariableop_88_adam_m_kf_attention_1_projection_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_89AssignVariableOp9assignvariableop_89_adam_v_kf_attention_1_projection_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_m_layer_normalization_3_gammaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_91AssignVariableOp6assignvariableop_91_adam_v_layer_normalization_3_gammaIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_92AssignVariableOp5assignvariableop_92_adam_m_layer_normalization_3_betaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_93AssignVariableOp5assignvariableop_93_adam_v_layer_normalization_3_betaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_m_conv1d_2_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_v_conv1d_2_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_m_conv1d_2_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_97AssignVariableOp(assignvariableop_97_adam_v_conv1d_2_biasIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adam_m_conv1d_3_kernelIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_v_conv1d_3_kernelIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_m_conv1d_3_biasIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adam_v_conv1d_3_biasIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_102AssignVariableOp7assignvariableop_102_adam_m_layer_normalization_4_gammaIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_103AssignVariableOp7assignvariableop_103_adam_v_layer_normalization_4_gammaIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_104AssignVariableOp6assignvariableop_104_adam_m_layer_normalization_4_betaIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_105AssignVariableOp6assignvariableop_105_adam_v_layer_normalization_4_betaIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_106AssignVariableOp7assignvariableop_106_adam_m_kf_attention_2_query_kernelIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_107AssignVariableOp7assignvariableop_107_adam_v_kf_attention_2_query_kernelIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_m_kf_attention_2_value_kernelIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_v_kf_attention_2_value_kernelIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:╫
AssignVariableOp_110AssignVariableOp<assignvariableop_110_adam_m_kf_attention_2_projection_kernelIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:╫
AssignVariableOp_111AssignVariableOp<assignvariableop_111_adam_v_kf_attention_2_projection_kernelIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:╒
AssignVariableOp_112AssignVariableOp:assignvariableop_112_adam_m_kf_attention_2_projection_biasIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:╒
AssignVariableOp_113AssignVariableOp:assignvariableop_113_adam_v_kf_attention_2_projection_biasIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_114AssignVariableOp7assignvariableop_114_adam_m_layer_normalization_5_gammaIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_115AssignVariableOp7assignvariableop_115_adam_v_layer_normalization_5_gammaIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_116AssignVariableOp6assignvariableop_116_adam_m_layer_normalization_5_betaIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_117AssignVariableOp6assignvariableop_117_adam_v_layer_normalization_5_betaIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_118AssignVariableOp+assignvariableop_118_adam_m_conv1d_4_kernelIdentity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_v_conv1d_4_kernelIdentity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_m_conv1d_4_biasIdentity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_121AssignVariableOp)assignvariableop_121_adam_v_conv1d_4_biasIdentity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_122AssignVariableOp+assignvariableop_122_adam_m_conv1d_5_kernelIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_123AssignVariableOp+assignvariableop_123_adam_v_conv1d_5_kernelIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_124AssignVariableOp)assignvariableop_124_adam_m_conv1d_5_biasIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_125AssignVariableOp)assignvariableop_125_adam_v_conv1d_5_biasIdentity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_126AssignVariableOp7assignvariableop_126_adam_m_layer_normalization_6_gammaIdentity_126:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_127AssignVariableOp7assignvariableop_127_adam_v_layer_normalization_6_gammaIdentity_127:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_128AssignVariableOp6assignvariableop_128_adam_m_layer_normalization_6_betaIdentity_128:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_129AssignVariableOp6assignvariableop_129_adam_v_layer_normalization_6_betaIdentity_129:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_130AssignVariableOp7assignvariableop_130_adam_m_kf_attention_3_query_kernelIdentity_130:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_131AssignVariableOp7assignvariableop_131_adam_v_kf_attention_3_query_kernelIdentity_131:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_132AssignVariableOp7assignvariableop_132_adam_m_kf_attention_3_value_kernelIdentity_132:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_133AssignVariableOp7assignvariableop_133_adam_v_kf_attention_3_value_kernelIdentity_133:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:╫
AssignVariableOp_134AssignVariableOp<assignvariableop_134_adam_m_kf_attention_3_projection_kernelIdentity_134:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:╫
AssignVariableOp_135AssignVariableOp<assignvariableop_135_adam_v_kf_attention_3_projection_kernelIdentity_135:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:╒
AssignVariableOp_136AssignVariableOp:assignvariableop_136_adam_m_kf_attention_3_projection_biasIdentity_136:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:╒
AssignVariableOp_137AssignVariableOp:assignvariableop_137_adam_v_kf_attention_3_projection_biasIdentity_137:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_138AssignVariableOp7assignvariableop_138_adam_m_layer_normalization_7_gammaIdentity_138:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_139AssignVariableOp7assignvariableop_139_adam_v_layer_normalization_7_gammaIdentity_139:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_140AssignVariableOp6assignvariableop_140_adam_m_layer_normalization_7_betaIdentity_140:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_141AssignVariableOp6assignvariableop_141_adam_v_layer_normalization_7_betaIdentity_141:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_142AssignVariableOp+assignvariableop_142_adam_m_conv1d_6_kernelIdentity_142:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_143AssignVariableOp+assignvariableop_143_adam_v_conv1d_6_kernelIdentity_143:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_144AssignVariableOp)assignvariableop_144_adam_m_conv1d_6_biasIdentity_144:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_145AssignVariableOp)assignvariableop_145_adam_v_conv1d_6_biasIdentity_145:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_146AssignVariableOp+assignvariableop_146_adam_m_conv1d_7_kernelIdentity_146:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_147AssignVariableOp+assignvariableop_147_adam_v_conv1d_7_kernelIdentity_147:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_148AssignVariableOp)assignvariableop_148_adam_m_conv1d_7_biasIdentity_148:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_149AssignVariableOp)assignvariableop_149_adam_v_conv1d_7_biasIdentity_149:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_150AssignVariableOp(assignvariableop_150_adam_m_dense_kernelIdentity_150:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_151AssignVariableOp(assignvariableop_151_adam_v_dense_kernelIdentity_151:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_152AssignVariableOp&assignvariableop_152_adam_m_dense_biasIdentity_152:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_153AssignVariableOp&assignvariableop_153_adam_v_dense_biasIdentity_153:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_154AssignVariableOp*assignvariableop_154_adam_m_dense_1_kernelIdentity_154:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_155AssignVariableOp*assignvariableop_155_adam_v_dense_1_kernelIdentity_155:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_156AssignVariableOp(assignvariableop_156_adam_m_dense_1_biasIdentity_156:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_157AssignVariableOp(assignvariableop_157_adam_v_dense_1_biasIdentity_157:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_158AssignVariableOpassignvariableop_158_total_1Identity_158:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_159AssignVariableOpassignvariableop_159_count_1Identity_159:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_160AssignVariableOpassignvariableop_160_totalIdentity_160:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_161AssignVariableOpassignvariableop_161_countIdentity_161:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ·
Identity_162Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_163IdentityIdentity_162:output:0^NoOp_1*
T0*
_output_shapes
: ц
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_163Identity_163:output:0*█
_input_shapes╔
╞: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
▌
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_530491

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
И
 
 kf_attention_1_cond_false_5312901
kf_attention_1_cond_531305:А1
kf_attention_1_cond_531307:А1
kf_attention_1_cond_531309:А(
kf_attention_1_cond_531311:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_531305kf_attention_1_cond_531307kf_attention_1_cond_531309kf_attention_1_cond_531311*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529977п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

ш
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_530000

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
И
 
 kf_attention_3_cond_false_5321081
kf_attention_3_cond_532123:А1
kf_attention_3_cond_532125:А1
kf_attention_3_cond_532127:А(
kf_attention_3_cond_532129:E
Akf_attention_3_cond_layer_normalization_6_statefulpartitionedcall 
kf_attention_3_cond_identityИв+kf_attention_3/cond/StatefulPartitionedCallТ
+kf_attention_3/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_3_cond_layer_normalization_6_statefulpartitionedcallkf_attention_3_cond_532123kf_attention_3_cond_532125kf_attention_3_cond_532127kf_attention_3_cond_532129*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_530363п
kf_attention_3/cond/IdentityIdentity4kf_attention_3/cond/StatefulPartitionedCall:output:0^kf_attention_3/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_3/cond/NoOpNoOp,^kf_attention_3/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_3_cond_identity%kf_attention_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_3/cond/StatefulPartitionedCall+kf_attention_3/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

И
 
 kf_attention_2_cond_false_5301111
kf_attention_2_cond_530171:А1
kf_attention_2_cond_530173:А1
kf_attention_2_cond_530175:А(
kf_attention_2_cond_530177:E
Akf_attention_2_cond_layer_normalization_4_statefulpartitionedcall 
kf_attention_2_cond_identityИв+kf_attention_2/cond/StatefulPartitionedCallТ
+kf_attention_2/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_2_cond_layer_normalization_4_statefulpartitionedcallkf_attention_2_cond_530171kf_attention_2_cond_530173kf_attention_2_cond_530175kf_attention_2_cond_530177*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_530170п
kf_attention_2/cond/IdentityIdentity4kf_attention_2/cond/StatefulPartitionedCall:output:0^kf_attention_2/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_2/cond/NoOpNoOp,^kf_attention_2/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_2_cond_identity%kf_attention_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_2/cond/StatefulPartitionedCall+kf_attention_2/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

ш
Я
6__inference_layer_normalization_6_layer_call_fn_533970

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_530297s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╓
Ъ
)__inference_conv1d_6_layer_call_fn_534129

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_530433s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
Р
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_533992

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ш
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_529865

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╓
Ъ
)__inference_conv1d_3_layer_call_fn_533711

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_530075s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ф
Э
4__inference_layer_normalization_layer_call_fn_533265

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_529718s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
┴
у
-__inference_kf_attention_layer_call_fn_533300

inputs
unknown:А 
	unknown_0:А 
	unknown_1:А
	unknown_2:
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529784s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╙
ь
kf_attention_cond_true_531232/
kf_attention_cond_531236:А/
kf_attention_cond_531238:А/
kf_attention_cond_531240:А&
kf_attention_cond_531242:A
=kf_attention_cond_layer_normalization_statefulpartitionedcall
kf_attention_cond_identityИв)kf_attention/cond/StatefulPartitionedCallВ
)kf_attention/cond/StatefulPartitionedCallStatefulPartitionedCall=kf_attention_cond_layer_normalization_statefulpartitionedcallkf_attention_cond_531236kf_attention_cond_531238kf_attention_cond_531240kf_attention_cond_531242*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_kf_attention_layer_call_and_return_conditional_losses_529750й
kf_attention/cond/IdentityIdentity2kf_attention/cond/StatefulPartitionedCall:output:0^kf_attention/cond/NoOp*
T0*+
_output_shapes
:         
Д
kf_attention/cond/NoOpNoOp*^kf_attention/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "A
kf_attention_cond_identity#kf_attention/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2V
)kf_attention/cond/StatefulPartitionedCall)kf_attention/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

и

e
F__inference_dropout_10_layer_call_and_return_conditional_losses_534089

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_530804

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ю
О
O__inference_layer_normalization_layer_call_and_return_conditional_losses_529718

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         
Л
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         
l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:л
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         
~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Е
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         
v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Б
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         
f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:         
А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ё
У
D__inference_conv1d_5_layer_call_and_return_conditional_losses_530268

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Л
╞
H__inference_kf_attention_layer_call_and_return_conditional_losses_529750

inputs<
%einsum_einsum_readvariableop_resource:А>
'einsum_1_einsum_readvariableop_resource:А>
'einsum_4_einsum_readvariableop_resource:А)
add_readvariableop_resource:
identityИвadd/ReadVariableOpвeinsum/Einsum/ReadVariableOpвeinsum_1/Einsum/ReadVariableOpвeinsum_4/Einsum/ReadVariableOpЗ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0к
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOЛ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0о
einsum_1/EinsumEinsuminputs&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC=
SqrtSqrtConst:output:0*
T0*
_output_shapes
: o
truedivRealDiveinsum/Einsum:output:0Sqrt:y:0*
T0*0
_output_shapes
:         
Аи
einsum_2/EinsumEinsumtruediv:z:0einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMf
SoftmaxSoftmaxeinsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

п
einsum_3/EinsumEinsumSoftmax:softmax:0einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIЛ
einsum_4/Einsum/ReadVariableOpReadVariableOp'einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0╗
einsum_4/EinsumEinsumeinsum_3/Einsum:output:0&einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0x
addAddV2einsum_4/Einsum:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         
╝
NoOpNoOp^add/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_4/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : : : 2(
add/ReadVariableOpadd/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_4/Einsum/ReadVariableOpeinsum_4/Einsum/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╚
У
D__inference_conv1d_4_layer_call_and_return_conditional_losses_533910

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:н
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         
Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
И
 
 kf_attention_1_cond_false_5299181
kf_attention_1_cond_529978:А1
kf_attention_1_cond_529980:А1
kf_attention_1_cond_529982:А(
kf_attention_1_cond_529984:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_529978kf_attention_1_cond_529980kf_attention_1_cond_529982kf_attention_1_cond_529984*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529977п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         

У

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_530648

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Эб
▌.
A__inference_model_layer_call_and_return_conditional_losses_533256

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:I
2kf_attention_einsum_einsum_readvariableop_resource:АK
4kf_attention_einsum_1_einsum_readvariableop_resource:АK
4kf_attention_einsum_4_einsum_readvariableop_resource:А6
(kf_attention_add_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:K
4kf_attention_1_einsum_einsum_readvariableop_resource:АM
6kf_attention_1_einsum_1_einsum_readvariableop_resource:АM
6kf_attention_1_einsum_4_einsum_readvariableop_resource:А8
*kf_attention_1_add_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_3_biasadd_readvariableop_resource:I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:E
7layer_normalization_4_batchnorm_readvariableop_resource:K
4kf_attention_2_einsum_einsum_readvariableop_resource:АM
6kf_attention_2_einsum_1_einsum_readvariableop_resource:АM
6kf_attention_2_einsum_4_einsum_readvariableop_resource:А8
*kf_attention_2_add_readvariableop_resource:I
;layer_normalization_5_batchnorm_mul_readvariableop_resource:E
7layer_normalization_5_batchnorm_readvariableop_resource:J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_4_biasadd_readvariableop_resource:J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_5_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:K
4kf_attention_3_einsum_einsum_readvariableop_resource:АM
6kf_attention_3_einsum_1_einsum_readvariableop_resource:АM
6kf_attention_3_einsum_4_einsum_readvariableop_resource:А8
*kf_attention_3_add_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:J
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_6_biasadd_readvariableop_resource:J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_7_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	
А4
%dense_biasadd_readvariableop_resource:	А:
&dense_1_matmul_readvariableop_resource:
А┘6
'dense_1_biasadd_readvariableop_resource:	┘
identityИвconv1d/BiasAdd/ReadVariableOpв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_1/BiasAdd/ReadVariableOpв+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_2/BiasAdd/ReadVariableOpв+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_3/BiasAdd/ReadVariableOpв+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_4/BiasAdd/ReadVariableOpв+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_5/BiasAdd/ReadVariableOpв+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_6/BiasAdd/ReadVariableOpв+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpвconv1d_7/BiasAdd/ReadVariableOpв+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвkf_attention/add/ReadVariableOpв)kf_attention/einsum/Einsum/ReadVariableOpв+kf_attention/einsum_1/Einsum/ReadVariableOpв+kf_attention/einsum_4/Einsum/ReadVariableOpв!kf_attention_1/add/ReadVariableOpв+kf_attention_1/einsum/Einsum/ReadVariableOpв-kf_attention_1/einsum_1/Einsum/ReadVariableOpв-kf_attention_1/einsum_4/Einsum/ReadVariableOpв!kf_attention_2/add/ReadVariableOpв+kf_attention_2/einsum/Einsum/ReadVariableOpв-kf_attention_2/einsum_1/Einsum/ReadVariableOpв-kf_attention_2/einsum_4/Einsum/ReadVariableOpв!kf_attention_3/add/ReadVariableOpв+kf_attention_3/einsum/Einsum/ReadVariableOpв-kf_attention_3/einsum_1/Einsum/ReadVariableOpв-kf_attention_3/einsum_4/Einsum/ReadVariableOpв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpв.layer_normalization_2/batchnorm/ReadVariableOpв2layer_normalization_2/batchnorm/mul/ReadVariableOpв.layer_normalization_3/batchnorm/ReadVariableOpв2layer_normalization_3/batchnorm/mul/ReadVariableOpв.layer_normalization_4/batchnorm/ReadVariableOpв2layer_normalization_4/batchnorm/mul/ReadVariableOpв.layer_normalization_5/batchnorm/ReadVariableOpв2layer_normalization_5/batchnorm/mul/ReadVariableOpв.layer_normalization_6/batchnorm/ReadVariableOpв2layer_normalization_6/batchnorm/mul/ReadVariableOpв.layer_normalization_7/batchnorm/ReadVariableOpв2layer_normalization_7/batchnorm/mul/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Щ
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         
│
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ч
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╜
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Й
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         
ж
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0┴
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
П
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
▓
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
Ю
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╜
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
▓
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
б
)kf_attention/einsum/Einsum/ReadVariableOpReadVariableOp2kf_attention_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0х
kf_attention/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:01kf_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOе
+kf_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp4kf_attention_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0щ
kf_attention/einsum_1/EinsumEinsum'layer_normalization/batchnorm/add_1:z:03kf_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOW
kf_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АCW
kf_attention/SqrtSqrtkf_attention/Const:output:0*
T0*
_output_shapes
: Ц
kf_attention/truedivRealDiv#kf_attention/einsum/Einsum:output:0kf_attention/Sqrt:y:0*
T0*0
_output_shapes
:         
А╧
kf_attention/einsum_2/EinsumEinsumkf_attention/truediv:z:0%kf_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMА
kf_attention/SoftmaxSoftmax%kf_attention/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

╓
kf_attention/einsum_3/EinsumEinsumkf_attention/Softmax:softmax:0%kf_attention/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIе
+kf_attention/einsum_4/Einsum/ReadVariableOpReadVariableOp4kf_attention_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0т
kf_attention/einsum_4/EinsumEinsum%kf_attention/einsum_3/Einsum:output:03kf_attention/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOД
kf_attention/add/ReadVariableOpReadVariableOp(kf_attention_add_readvariableop_resource*
_output_shapes
:*
dtype0Я
kf_attention/addAddV2%kf_attention/einsum_4/Einsum:output:0'kf_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?К
dropout_1/dropout/MulMulkf_attention/add:z:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:         
[
dropout_1/dropout/ShapeShapekf_attention/add:z:0*
T0*
_output_shapes
:д
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
Ж
tf.__operators__.add/AddV2AddV2#dropout_1/dropout/SelectV2:output:0inputs*
T0*+
_output_shapes
:         
~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╨
"layer_normalization_1/moments/meanMeantf.__operators__.add/AddV2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         
╧
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetf.__operators__.add/AddV2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
л
%layer_normalization_1/batchnorm/mul_1Multf.__operators__.add/AddV2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ▓
conv1d/Conv1D/ExpandDims
ExpandDims)layer_normalization_1/batchnorm/add_1:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┬
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        А
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ц
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:         
\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?П
dropout_2/dropout/MulMulconv1d/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:         
`
dropout_2/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:д
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
^
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_2/dropout/SelectV2SelectV2"dropout_2/dropout/GreaterEqual:z:0dropout_2/dropout/Mul:z:0"dropout_2/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ░
conv1d_1/Conv1D/ExpandDims
ExpandDims#dropout_2/dropout/SelectV2:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ц
tf.__operators__.add_1/AddV2AddV2conv1d_1/BiasAdd:output:0tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_2/moments/meanMean tf.__operators__.add_1/AddV2:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_2/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_1/AddV2:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_2/batchnorm/mul_1Mul tf.__operators__.add_1/AddV2:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
е
+kf_attention_1/einsum/Einsum/ReadVariableOpReadVariableOp4kf_attention_1_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ы
kf_attention_1/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:03kf_attention_1/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOй
-kf_attention_1/einsum_1/Einsum/ReadVariableOpReadVariableOp6kf_attention_1_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0я
kf_attention_1/einsum_1/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:05kf_attention_1/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOY
kf_attention_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC[
kf_attention_1/SqrtSqrtkf_attention_1/Const:output:0*
T0*
_output_shapes
: Ь
kf_attention_1/truedivRealDiv%kf_attention_1/einsum/Einsum:output:0kf_attention_1/Sqrt:y:0*
T0*0
_output_shapes
:         
А╒
kf_attention_1/einsum_2/EinsumEinsumkf_attention_1/truediv:z:0'kf_attention_1/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMД
kf_attention_1/SoftmaxSoftmax'kf_attention_1/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

▄
kf_attention_1/einsum_3/EinsumEinsum kf_attention_1/Softmax:softmax:0'kf_attention_1/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIй
-kf_attention_1/einsum_4/Einsum/ReadVariableOpReadVariableOp6kf_attention_1_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ш
kf_attention_1/einsum_4/EinsumEinsum'kf_attention_1/einsum_3/Einsum:output:05kf_attention_1/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOИ
!kf_attention_1/add/ReadVariableOpReadVariableOp*kf_attention_1_add_readvariableop_resource*
_output_shapes
:*
dtype0е
kf_attention_1/addAddV2'kf_attention_1/einsum_4/Einsum:output:0)kf_attention_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?М
dropout_4/dropout/MulMulkf_attention_1/add:z:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:         
]
dropout_4/dropout/ShapeShapekf_attention_1/add:z:0*
T0*
_output_shapes
:д
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
^
dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_4/dropout/SelectV2SelectV2"dropout_4/dropout/GreaterEqual:z:0dropout_4/dropout/Mul:z:0"dropout_4/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
в
tf.__operators__.add_2/AddV2AddV2#dropout_4/dropout/SelectV2:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_3/moments/meanMean tf.__operators__.add_2/AddV2:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_2/AddV2:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_3/batchnorm/mul_1Mul tf.__operators__.add_2/AddV2:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╢
conv1d_2/Conv1D/ExpandDims
ExpandDims)layer_normalization_3/batchnorm/add_1:z:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:         
\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?С
dropout_5/dropout/MulMulconv1d_2/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:         
b
dropout_5/dropout/ShapeShapeconv1d_2/Relu:activations:0*
T0*
_output_shapes
:д
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ░
conv1d_3/Conv1D/ExpandDims
ExpandDims#dropout_5/dropout/SelectV2:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ш
tf.__operators__.add_3/AddV2AddV2conv1d_3/BiasAdd:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_4/moments/meanMean tf.__operators__.add_3/AddV2:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_4/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_3/AddV2:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_4/batchnorm/mul_1Mul tf.__operators__.add_3/AddV2:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
е
+kf_attention_2/einsum/Einsum/ReadVariableOpReadVariableOp4kf_attention_2_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ы
kf_attention_2/einsum/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:03kf_attention_2/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOй
-kf_attention_2/einsum_1/Einsum/ReadVariableOpReadVariableOp6kf_attention_2_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0я
kf_attention_2/einsum_1/EinsumEinsum)layer_normalization_4/batchnorm/add_1:z:05kf_attention_2/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOY
kf_attention_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC[
kf_attention_2/SqrtSqrtkf_attention_2/Const:output:0*
T0*
_output_shapes
: Ь
kf_attention_2/truedivRealDiv%kf_attention_2/einsum/Einsum:output:0kf_attention_2/Sqrt:y:0*
T0*0
_output_shapes
:         
А╒
kf_attention_2/einsum_2/EinsumEinsumkf_attention_2/truediv:z:0'kf_attention_2/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMД
kf_attention_2/SoftmaxSoftmax'kf_attention_2/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

▄
kf_attention_2/einsum_3/EinsumEinsum kf_attention_2/Softmax:softmax:0'kf_attention_2/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIй
-kf_attention_2/einsum_4/Einsum/ReadVariableOpReadVariableOp6kf_attention_2_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ш
kf_attention_2/einsum_4/EinsumEinsum'kf_attention_2/einsum_3/Einsum:output:05kf_attention_2/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOИ
!kf_attention_2/add/ReadVariableOpReadVariableOp*kf_attention_2_add_readvariableop_resource*
_output_shapes
:*
dtype0е
kf_attention_2/addAddV2'kf_attention_2/einsum_4/Einsum:output:0)kf_attention_2/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?М
dropout_7/dropout/MulMulkf_attention_2/add:z:0 dropout_7/dropout/Const:output:0*
T0*+
_output_shapes
:         
]
dropout_7/dropout/ShapeShapekf_attention_2/add:z:0*
T0*
_output_shapes
:д
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
^
dropout_7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_7/dropout/SelectV2SelectV2"dropout_7/dropout/GreaterEqual:z:0dropout_7/dropout/Mul:z:0"dropout_7/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
в
tf.__operators__.add_4/AddV2AddV2#dropout_7/dropout/SelectV2:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_5/moments/meanMean tf.__operators__.add_4/AddV2:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_4/AddV2:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_5/batchnorm/mul_1Mul tf.__operators__.add_4/AddV2:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
i
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╢
conv1d_4/Conv1D/ExpandDims
ExpandDims)layer_normalization_5/batchnorm/add_1:z:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
f
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:         
\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?С
dropout_8/dropout/MulMulconv1d_4/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:         
b
dropout_8/dropout/ShapeShapeconv1d_4/Relu:activations:0*
T0*
_output_shapes
:д
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
^
dropout_8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┐
dropout_8/dropout/SelectV2SelectV2"dropout_8/dropout/GreaterEqual:z:0dropout_8/dropout/Mul:z:0"dropout_8/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ░
conv1d_5/Conv1D/ExpandDims
ExpandDims#dropout_8/dropout/SelectV2:output:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ш
tf.__operators__.add_5/AddV2AddV2conv1d_5/BiasAdd:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_6/moments/meanMean tf.__operators__.add_5/AddV2:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_6/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_5/AddV2:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_6/batchnorm/mul_1Mul tf.__operators__.add_5/AddV2:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
е
+kf_attention_3/einsum/Einsum/ReadVariableOpReadVariableOp4kf_attention_3_einsum_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ы
kf_attention_3/einsum/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:03kf_attention_3/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...NI,HIO->...NHOй
-kf_attention_3/einsum_1/Einsum/ReadVariableOpReadVariableOp6kf_attention_3_einsum_1_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0я
kf_attention_3/einsum_1/EinsumEinsum)layer_normalization_6/batchnorm/add_1:z:05kf_attention_3/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         
А*
equation...MI,HIO->...MHOY
kf_attention_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  АC[
kf_attention_3/SqrtSqrtkf_attention_3/Const:output:0*
T0*
_output_shapes
: Ь
kf_attention_3/truedivRealDiv%kf_attention_3/einsum/Einsum:output:0kf_attention_3/Sqrt:y:0*
T0*0
_output_shapes
:         
А╒
kf_attention_3/einsum_2/EinsumEinsumkf_attention_3/truediv:z:0'kf_attention_3/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:         

*#
equation...NHO,...MHO->...HNMД
kf_attention_3/SoftmaxSoftmax'kf_attention_3/einsum_2/Einsum:output:0*
T0*/
_output_shapes
:         

▄
kf_attention_3/einsum_3/EinsumEinsum kf_attention_3/Softmax:softmax:0'kf_attention_3/einsum_1/Einsum:output:0*
N*
T0*0
_output_shapes
:         
А*#
equation...HNM,...MHI->...NHIй
-kf_attention_3/einsum_4/Einsum/ReadVariableOpReadVariableOp6kf_attention_3_einsum_4_einsum_readvariableop_resource*#
_output_shapes
:А*
dtype0ш
kf_attention_3/einsum_4/EinsumEinsum'kf_attention_3/einsum_3/Einsum:output:05kf_attention_3/einsum_4/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         
*
equation...NHI,HIO->...NOИ
!kf_attention_3/add/ReadVariableOpReadVariableOp*kf_attention_3_add_readvariableop_resource*
_output_shapes
:*
dtype0е
kf_attention_3/addAddV2'kf_attention_3/einsum_4/Einsum:output:0)kf_attention_3/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?О
dropout_10/dropout/MulMulkf_attention_3/add:z:0!dropout_10/dropout/Const:output:0*
T0*+
_output_shapes
:         
^
dropout_10/dropout/ShapeShapekf_attention_3/add:z:0*
T0*
_output_shapes
:ж
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╦
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
_
dropout_10/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
dropout_10/dropout/SelectV2SelectV2#dropout_10/dropout/GreaterEqual:z:0dropout_10/dropout/Mul:z:0#dropout_10/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
г
tf.__operators__.add_6/AddV2AddV2$dropout_10/dropout/SelectV2:output:0 tf.__operators__.add_5/AddV2:z:0*
T0*+
_output_shapes
:         
~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╥
"layer_normalization_7/moments/meanMean tf.__operators__.add_6/AddV2:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(Э
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:         
╤
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_6/AddV2:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:         
В
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         
Н
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:         
к
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╟
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
н
%layer_normalization_7/batchnorm/mul_1Mul tf.__operators__.add_6/AddV2:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:         
в
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0├
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         
╕
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:         
i
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ╢
conv1d_6/Conv1D/ExpandDims
ExpandDims)layer_normalization_7/batchnorm/add_1:z:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_6/BiasAddBiasAdd conv1d_6/Conv1D/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
f
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*+
_output_shapes
:         
]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?У
dropout_11/dropout/MulMulconv1d_6/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*+
_output_shapes
:         
c
dropout_11/dropout/ShapeShapeconv1d_6/Relu:activations:0*
T0*
_output_shapes
:ж
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╦
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
_
dropout_11/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ├
dropout_11/dropout/SelectV2SelectV2#dropout_11/dropout/GreaterEqual:z:0dropout_11/dropout/Mul:z:0#dropout_11/dropout/Const_1:output:0*
T0*+
_output_shapes
:         
i
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ▒
conv1d_7/Conv1D/ExpandDims
ExpandDims$dropout_11/dropout/SelectV2:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
д
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╗
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╚
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
Т
conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        Д
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv1d_7/BiasAddBiasAdd conv1d_7/Conv1D/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
Ш
tf.__operators__.add_7/AddV2AddV2conv1d_7/BiasAdd:output:0 tf.__operators__.add_6/AddV2:z:0*
T0*+
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :│
global_average_pooling1d/MeanMean tf.__operators__.add_7/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         
Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	
А*
dtype0Ц
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         А]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?Н
dropout_12/dropout/MulMuldense/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*(
_output_shapes
:         А`
dropout_12/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:г
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>╚
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А_
dropout_12/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    └
dropout_12/dropout/SelectV2SelectV2#dropout_12/dropout/GreaterEqual:z:0dropout_12/dropout/Mul:z:0#dropout_12/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АЖ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
А┘*
dtype0Ш
dense_1/MatMulMatMul$dropout_12/dropout/SelectV2:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘Г
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:┘*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┘g
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ┘i
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ┘м
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^kf_attention/add/ReadVariableOp*^kf_attention/einsum/Einsum/ReadVariableOp,^kf_attention/einsum_1/Einsum/ReadVariableOp,^kf_attention/einsum_4/Einsum/ReadVariableOp"^kf_attention_1/add/ReadVariableOp,^kf_attention_1/einsum/Einsum/ReadVariableOp.^kf_attention_1/einsum_1/Einsum/ReadVariableOp.^kf_attention_1/einsum_4/Einsum/ReadVariableOp"^kf_attention_2/add/ReadVariableOp,^kf_attention_2/einsum/Einsum/ReadVariableOp.^kf_attention_2/einsum_1/Einsum/ReadVariableOp.^kf_attention_2/einsum_4/Einsum/ReadVariableOp"^kf_attention_3/add/ReadVariableOp,^kf_attention_3/einsum/Einsum/ReadVariableOp.^kf_attention_3/einsum_1/Einsum/ReadVariableOp.^kf_attention_3/einsum_4/Einsum/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:         
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
kf_attention/add/ReadVariableOpkf_attention/add/ReadVariableOp2V
)kf_attention/einsum/Einsum/ReadVariableOp)kf_attention/einsum/Einsum/ReadVariableOp2Z
+kf_attention/einsum_1/Einsum/ReadVariableOp+kf_attention/einsum_1/Einsum/ReadVariableOp2Z
+kf_attention/einsum_4/Einsum/ReadVariableOp+kf_attention/einsum_4/Einsum/ReadVariableOp2F
!kf_attention_1/add/ReadVariableOp!kf_attention_1/add/ReadVariableOp2Z
+kf_attention_1/einsum/Einsum/ReadVariableOp+kf_attention_1/einsum/Einsum/ReadVariableOp2^
-kf_attention_1/einsum_1/Einsum/ReadVariableOp-kf_attention_1/einsum_1/Einsum/ReadVariableOp2^
-kf_attention_1/einsum_4/Einsum/ReadVariableOp-kf_attention_1/einsum_4/Einsum/ReadVariableOp2F
!kf_attention_2/add/ReadVariableOp!kf_attention_2/add/ReadVariableOp2Z
+kf_attention_2/einsum/Einsum/ReadVariableOp+kf_attention_2/einsum/Einsum/ReadVariableOp2^
-kf_attention_2/einsum_1/Einsum/ReadVariableOp-kf_attention_2/einsum_1/Einsum/ReadVariableOp2^
-kf_attention_2/einsum_4/Einsum/ReadVariableOp-kf_attention_2/einsum_4/Einsum/ReadVariableOp2F
!kf_attention_3/add/ReadVariableOp!kf_attention_3/add/ReadVariableOp2Z
+kf_attention_3/einsum/Einsum/ReadVariableOp+kf_attention_3/einsum/Einsum/ReadVariableOp2^
-kf_attention_3/einsum_1/Einsum/ReadVariableOp-kf_attention_3/einsum_1/Einsum/ReadVariableOp2^
-kf_attention_3/einsum_4/Einsum/ReadVariableOp-kf_attention_3/einsum_4/Einsum/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
з

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_530917

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         
T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:         
e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
З
■
kf_attention_1_cond_true_5317491
kf_attention_1_cond_531753:А1
kf_attention_1_cond_531755:А1
kf_attention_1_cond_531757:А(
kf_attention_1_cond_531759:E
Akf_attention_1_cond_layer_normalization_2_statefulpartitionedcall 
kf_attention_1_cond_identityИв+kf_attention_1/cond/StatefulPartitionedCallТ
+kf_attention_1/cond/StatefulPartitionedCallStatefulPartitionedCallAkf_attention_1_cond_layer_normalization_2_statefulpartitionedcallkf_attention_1_cond_531753kf_attention_1_cond_531755kf_attention_1_cond_531757kf_attention_1_cond_531759*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_529943п
kf_attention_1/cond/IdentityIdentity4kf_attention_1/cond/StatefulPartitionedCall:output:0^kf_attention_1/cond/NoOp*
T0*+
_output_shapes
:         
И
kf_attention_1/cond/NoOpNoOp,^kf_attention_1/cond/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "E
kf_attention_1_cond_identity%kf_attention_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : :         
2Z
+kf_attention_1/cond/StatefulPartitionedCall+kf_attention_1/cond/StatefulPartitionedCall:1-
+
_output_shapes
:         
"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*п
serving_defaultЫ
?
input_14
serving_default_input_1:0         
<
dense_11
StatefulPartitionedCall:0         ┘tensorflow/serving/predict:№┐
▀

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
layer-25
layer_with_weights-14
layer-26
layer-27
layer_with_weights-15
layer-28
layer_with_weights-16
layer-29
layer-30
 layer-31
!layer_with_weights-17
!layer-32
"layer_with_weights-18
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer-39
)layer_with_weights-21
)layer-40
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_default_save_signature
1	optimizer
2
signatures"
_tf_keras_network
"
_tf_keras_input_layer
─
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9axis
	:gamma
;beta"
_tf_keras_layer
В
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
Bdropout
Cquery_kernel
Dvalue_kernel
Eprojection_kernel
Fprojection_bias"
_tf_keras_layer
╝
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator"
_tf_keras_layer
(
N	keras_api"
_tf_keras_layer
─
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta"
_tf_keras_layer
▌
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias
 `_jit_compiled_convolution_op"
_tf_keras_layer
╝
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator"
_tf_keras_layer
▌
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias
 p_jit_compiled_convolution_op"
_tf_keras_layer
(
q	keras_api"
_tf_keras_layer
─
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xaxis
	ygamma
zbeta"
_tf_keras_layer
И
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses
Бdropout
Вquery_kernel
Гvalue_kernel
Дprojection_kernel
Еprojection_bias"
_tf_keras_layer
├
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses
М_random_generator"
_tf_keras_layer
)
Н	keras_api"
_tf_keras_layer
═
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses
	Фaxis

Хgamma
	Цbeta"
_tf_keras_layer
ц
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
Эkernel
	Юbias
!Я_jit_compiled_convolution_op"
_tf_keras_layer
├
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
ж_random_generator"
_tf_keras_layer
ц
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
нkernel
	оbias
!п_jit_compiled_convolution_op"
_tf_keras_layer
)
░	keras_api"
_tf_keras_layer
═
▒	variables
▓trainable_variables
│regularization_losses
┤	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses
	╖axis

╕gamma
	╣beta"
_tf_keras_layer
Н
║	variables
╗trainable_variables
╝regularization_losses
╜	keras_api
╛__call__
+┐&call_and_return_all_conditional_losses
└dropout
┴query_kernel
┬value_kernel
├projection_kernel
─projection_bias"
_tf_keras_layer
├
┼	variables
╞trainable_variables
╟regularization_losses
╚	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses
╦_random_generator"
_tf_keras_layer
)
╠	keras_api"
_tf_keras_layer
═
═	variables
╬trainable_variables
╧regularization_losses
╨	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses
	╙axis

╘gamma
	╒beta"
_tf_keras_layer
ц
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses
▄kernel
	▌bias
!▐_jit_compiled_convolution_op"
_tf_keras_layer
├
▀	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
х_random_generator"
_tf_keras_layer
ц
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses
ьkernel
	эbias
!ю_jit_compiled_convolution_op"
_tf_keras_layer
)
я	keras_api"
_tf_keras_layer
═
Ё	variables
ёtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses
	Ўaxis

ўgamma
	°beta"
_tf_keras_layer
Н
∙	variables
·trainable_variables
√regularization_losses
№	keras_api
¤__call__
+■&call_and_return_all_conditional_losses
 dropout
Аquery_kernel
Бvalue_kernel
Вprojection_kernel
Гprojection_bias"
_tf_keras_layer
├
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
К_random_generator"
_tf_keras_layer
)
Л	keras_api"
_tf_keras_layer
═
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
	Тaxis

Уgamma
	Фbeta"
_tf_keras_layer
ц
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias
!Э_jit_compiled_convolution_op"
_tf_keras_layer
├
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses
д_random_generator"
_tf_keras_layer
ц
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
лkernel
	мbias
!н_jit_compiled_convolution_op"
_tf_keras_layer
)
о	keras_api"
_tf_keras_layer
л
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗kernel
	╝bias"
_tf_keras_layer
├
╜	variables
╛trainable_variables
┐regularization_losses
└	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses
├_random_generator"
_tf_keras_layer
├
─	variables
┼trainable_variables
╞regularization_losses
╟	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses
╩kernel
	╦bias"
_tf_keras_layer
▄
:0
;1
C2
D3
E4
F5
V6
W7
^8
_9
n10
o11
y12
z13
В14
Г15
Д16
Е17
Х18
Ц19
Э20
Ю21
н22
о23
╕24
╣25
┴26
┬27
├28
─29
╘30
╒31
▄32
▌33
ь34
э35
ў36
°37
А38
Б39
В40
Г41
У42
Ф43
Ы44
Ь45
л46
м47
╗48
╝49
╩50
╦51"
trackable_list_wrapper
▄
:0
;1
C2
D3
E4
F5
V6
W7
^8
_9
n10
o11
y12
z13
В14
Г15
Д16
Е17
Х18
Ц19
Э20
Ю21
н22
о23
╕24
╣25
┴26
┬27
├28
─29
╘30
╒31
▄32
▌33
ь34
э35
ў36
°37
А38
Б39
В40
Г41
У42
Ф43
Ы44
Ь45
л46
м47
╗48
╝49
╩50
╦51"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
╒
╤trace_0
╥trace_1
╙trace_2
╘trace_32т
&__inference_model_layer_call_fn_530618
&__inference_model_layer_call_fn_532394
&__inference_model_layer_call_fn_532503
&__inference_model_layer_call_fn_531684┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0z╥trace_1z╙trace_2z╘trace_3
┴
╒trace_0
╓trace_1
╫trace_2
╪trace_32╬
A__inference_model_layer_call_and_return_conditional_losses_532848
A__inference_model_layer_call_and_return_conditional_losses_533256
A__inference_model_layer_call_and_return_conditional_losses_531928
A__inference_model_layer_call_and_return_conditional_losses_532172┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╒trace_0z╓trace_1z╫trace_2z╪trace_3
╠B╔
!__inference__wrapped_model_529676input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
┘
_variables
┌_iterations
█_learning_rate
▄_index_dict
▌
_momentums
▐_velocities
▀_update_step_xla"
experimentalOptimizer
-
рserving_default"
signature_map
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
·
цtrace_02█
4__inference_layer_normalization_layer_call_fn_533265в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0
Х
чtrace_02Ў
O__inference_layer_normalization_layer_call_and_return_conditional_losses_533287в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
 "
trackable_list_wrapper
':%2layer_normalization/gamma
&:$2layer_normalization/beta
<
C0
D1
E2
F3"
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
█
эtrace_0
юtrace_12а
-__inference_kf_attention_layer_call_fn_533300
-__inference_kf_attention_layer_call_fn_533313┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zэtrace_0zюtrace_1
С
яtrace_0
Ёtrace_12╓
H__inference_kf_attention_layer_call_and_return_conditional_losses_533335
H__inference_kf_attention_layer_call_and_return_conditional_losses_533357┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0zЁtrace_1
A
ё	keras_api
Є_random_generator"
_tf_keras_layer
0:.А2kf_attention/query_kernel
0:.А2kf_attention/value_kernel
5:3А2kf_attention/projection_kernel
*:(2kf_attention/projection_bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
єnon_trainable_variables
Їlayers
їmetrics
 Ўlayer_regularization_losses
ўlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
╔
°trace_0
∙trace_12О
*__inference_dropout_1_layer_call_fn_533362
*__inference_dropout_1_layer_call_fn_533367│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0z∙trace_1
 
·trace_0
√trace_12─
E__inference_dropout_1_layer_call_and_return_conditional_losses_533372
E__inference_dropout_1_layer_call_and_return_conditional_losses_533384│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z·trace_0z√trace_1
"
_generic_user_object
"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
№non_trainable_variables
¤layers
■metrics
  layer_regularization_losses
Аlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
№
Бtrace_02▌
6__inference_layer_normalization_1_layer_call_fn_533393в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zБtrace_0
Ч
Вtrace_02°
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_533415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_1/gamma
(:&2layer_normalization_1/beta
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
э
Иtrace_02╬
'__inference_conv1d_layer_call_fn_533424в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zИtrace_0
И
Йtrace_02щ
B__inference_conv1d_layer_call_and_return_conditional_losses_533440в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЙtrace_0
#:!2conv1d/kernel
:2conv1d/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
╔
Пtrace_0
Рtrace_12О
*__inference_dropout_2_layer_call_fn_533445
*__inference_dropout_2_layer_call_fn_533450│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zПtrace_0zРtrace_1
 
Сtrace_0
Тtrace_12─
E__inference_dropout_2_layer_call_and_return_conditional_losses_533455
E__inference_dropout_2_layer_call_and_return_conditional_losses_533467│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0zТtrace_1
"
_generic_user_object
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
я
Шtrace_02╨
)__inference_conv1d_1_layer_call_fn_533476в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
К
Щtrace_02ы
D__inference_conv1d_1_layer_call_and_return_conditional_losses_533491в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЩtrace_0
%:#2conv1d_1/kernel
:2conv1d_1/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
"
_generic_user_object
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
№
Яtrace_02▌
6__inference_layer_normalization_2_layer_call_fn_533500в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЯtrace_0
Ч
аtrace_02°
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_533522в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_2/gamma
(:&2layer_normalization_2/beta
@
В0
Г1
Д2
Е3"
trackable_list_wrapper
@
В0
Г1
Д2
Е3"
trackable_list_wrapper
 "
trackable_list_wrapper
┤
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
▀
жtrace_0
зtrace_12д
/__inference_kf_attention_1_layer_call_fn_533535
/__inference_kf_attention_1_layer_call_fn_533548┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0zзtrace_1
Х
иtrace_0
йtrace_12┌
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533570
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533592┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0zйtrace_1
A
к	keras_api
л_random_generator"
_tf_keras_layer
2:0А2kf_attention_1/query_kernel
2:0А2kf_attention_1/value_kernel
7:5А2 kf_attention_1/projection_kernel
,:*2kf_attention_1/projection_bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
╔
▒trace_0
▓trace_12О
*__inference_dropout_4_layer_call_fn_533597
*__inference_dropout_4_layer_call_fn_533602│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0z▓trace_1
 
│trace_0
┤trace_12─
E__inference_dropout_4_layer_call_and_return_conditional_losses_533607
E__inference_dropout_4_layer_call_and_return_conditional_losses_533619│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0z┤trace_1
"
_generic_user_object
"
_generic_user_object
0
Х0
Ц1"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
№
║trace_02▌
6__inference_layer_normalization_3_layer_call_fn_533628в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0
Ч
╗trace_02°
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_533650в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0
 "
trackable_list_wrapper
):'2layer_normalization_3/gamma
(:&2layer_normalization_3/beta
0
Э0
Ю1"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
я
┴trace_02╨
)__inference_conv1d_2_layer_call_fn_533659в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0
К
┬trace_02ы
D__inference_conv1d_2_layer_call_and_return_conditional_losses_533675в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0
%:#2conv1d_2/kernel
:2conv1d_2/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
╔
╚trace_0
╔trace_12О
*__inference_dropout_5_layer_call_fn_533680
*__inference_dropout_5_layer_call_fn_533685│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╚trace_0z╔trace_1
 
╩trace_0
╦trace_12─
E__inference_dropout_5_layer_call_and_return_conditional_losses_533690
E__inference_dropout_5_layer_call_and_return_conditional_losses_533702│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0z╦trace_1
"
_generic_user_object
0
н0
о1"
trackable_list_wrapper
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
я
╤trace_02╨
)__inference_conv1d_3_layer_call_fn_533711в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
К
╥trace_02ы
D__inference_conv1d_3_layer_call_and_return_conditional_losses_533726в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
%:#2conv1d_3/kernel
:2conv1d_3/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
"
_generic_user_object
0
╕0
╣1"
trackable_list_wrapper
0
╕0
╣1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
▒	variables
▓trainable_variables
│regularization_losses
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
№
╪trace_02▌
6__inference_layer_normalization_4_layer_call_fn_533735в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
Ч
┘trace_02°
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_533757в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
 "
trackable_list_wrapper
):'2layer_normalization_4/gamma
(:&2layer_normalization_4/beta
@
┴0
┬1
├2
─3"
trackable_list_wrapper
@
┴0
┬1
├2
─3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
║	variables
╗trainable_variables
╝regularization_losses
╛__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
▀
▀trace_0
рtrace_12д
/__inference_kf_attention_2_layer_call_fn_533770
/__inference_kf_attention_2_layer_call_fn_533783┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0zрtrace_1
Х
сtrace_0
тtrace_12┌
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533805
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533827┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0zтtrace_1
A
у	keras_api
ф_random_generator"
_tf_keras_layer
2:0А2kf_attention_2/query_kernel
2:0А2kf_attention_2/value_kernel
7:5А2 kf_attention_2/projection_kernel
,:*2kf_attention_2/projection_bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
┼	variables
╞trainable_variables
╟regularization_losses
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
╔
ъtrace_0
ыtrace_12О
*__inference_dropout_7_layer_call_fn_533832
*__inference_dropout_7_layer_call_fn_533837│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zъtrace_0zыtrace_1
 
ьtrace_0
эtrace_12─
E__inference_dropout_7_layer_call_and_return_conditional_losses_533842
E__inference_dropout_7_layer_call_and_return_conditional_losses_533854│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zьtrace_0zэtrace_1
"
_generic_user_object
"
_generic_user_object
0
╘0
╒1"
trackable_list_wrapper
0
╘0
╒1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
юnon_trainable_variables
яlayers
Ёmetrics
 ёlayer_regularization_losses
Єlayer_metrics
═	variables
╬trainable_variables
╧regularization_losses
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
№
єtrace_02▌
6__inference_layer_normalization_5_layer_call_fn_533863в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zєtrace_0
Ч
Їtrace_02°
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_533885в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЇtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_5/gamma
(:&2layer_normalization_5/beta
0
▄0
▌1"
trackable_list_wrapper
0
▄0
▌1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
їnon_trainable_variables
Ўlayers
ўmetrics
 °layer_regularization_losses
∙layer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
я
·trace_02╨
)__inference_conv1d_4_layer_call_fn_533894в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z·trace_0
К
√trace_02ы
D__inference_conv1d_4_layer_call_and_return_conditional_losses_533910в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z√trace_0
%:#2conv1d_4/kernel
:2conv1d_4/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
№non_trainable_variables
¤layers
■metrics
  layer_regularization_losses
Аlayer_metrics
▀	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
╔
Бtrace_0
Вtrace_12О
*__inference_dropout_8_layer_call_fn_533915
*__inference_dropout_8_layer_call_fn_533920│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zБtrace_0zВtrace_1
 
Гtrace_0
Дtrace_12─
E__inference_dropout_8_layer_call_and_return_conditional_losses_533925
E__inference_dropout_8_layer_call_and_return_conditional_losses_533937│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0zДtrace_1
"
_generic_user_object
0
ь0
э1"
trackable_list_wrapper
0
ь0
э1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
я
Кtrace_02╨
)__inference_conv1d_5_layer_call_fn_533946в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
К
Лtrace_02ы
D__inference_conv1d_5_layer_call_and_return_conditional_losses_533961в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЛtrace_0
%:#2conv1d_5/kernel
:2conv1d_5/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
"
_generic_user_object
0
ў0
°1"
trackable_list_wrapper
0
ў0
°1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
Ё	variables
ёtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
№
Сtrace_02▌
6__inference_layer_normalization_6_layer_call_fn_533970в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
Ч
Тtrace_02°
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_533992в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zТtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_6/gamma
(:&2layer_normalization_6/beta
@
А0
Б1
В2
Г3"
trackable_list_wrapper
@
А0
Б1
В2
Г3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
∙	variables
·trainable_variables
√regularization_losses
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
▀
Шtrace_0
Щtrace_12д
/__inference_kf_attention_3_layer_call_fn_534005
/__inference_kf_attention_3_layer_call_fn_534018┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0zЩtrace_1
Х
Ъtrace_0
Ыtrace_12┌
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534040
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534062┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0zЫtrace_1
A
Ь	keras_api
Э_random_generator"
_tf_keras_layer
2:0А2kf_attention_3/query_kernel
2:0А2kf_attention_3/value_kernel
7:5А2 kf_attention_3/projection_kernel
,:*2kf_attention_3/projection_bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
╦
гtrace_0
дtrace_12Р
+__inference_dropout_10_layer_call_fn_534067
+__inference_dropout_10_layer_call_fn_534072│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zгtrace_0zдtrace_1
Б
еtrace_0
жtrace_12╞
F__inference_dropout_10_layer_call_and_return_conditional_losses_534077
F__inference_dropout_10_layer_call_and_return_conditional_losses_534089│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0zжtrace_1
"
_generic_user_object
"
_generic_user_object
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
№
мtrace_02▌
6__inference_layer_normalization_7_layer_call_fn_534098в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0
Ч
нtrace_02°
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_534120в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zнtrace_0
 "
trackable_list_wrapper
):'2layer_normalization_7/gamma
(:&2layer_normalization_7/beta
0
Ы0
Ь1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
я
│trace_02╨
)__inference_conv1d_6_layer_call_fn_534129в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
К
┤trace_02ы
D__inference_conv1d_6_layer_call_and_return_conditional_losses_534145в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0
%:#2conv1d_6/kernel
:2conv1d_6/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
╦
║trace_0
╗trace_12Р
+__inference_dropout_11_layer_call_fn_534150
+__inference_dropout_11_layer_call_fn_534155│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0z╗trace_1
Б
╝trace_0
╜trace_12╞
F__inference_dropout_11_layer_call_and_return_conditional_losses_534160
F__inference_dropout_11_layer_call_and_return_conditional_losses_534172│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╝trace_0z╜trace_1
"
_generic_user_object
0
л0
м1"
trackable_list_wrapper
0
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
я
├trace_02╨
)__inference_conv1d_7_layer_call_fn_534181в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0
К
─trace_02ы
D__inference_conv1d_7_layer_call_and_return_conditional_losses_534196в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
%:#2conv1d_7/kernel
:2conv1d_7/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
п	variables
░trainable_variables
▒regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
М
╩trace_02э
9__inference_global_average_pooling1d_layer_call_fn_534201п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
з
╦trace_02И
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_534207п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
0
╗0
╝1"
trackable_list_wrapper
0
╗0
╝1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
ь
╤trace_02═
&__inference_dense_layer_call_fn_534216в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
З
╥trace_02ш
A__inference_dense_layer_call_and_return_conditional_losses_534227в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
:	
А2dense/kernel
:А2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
╜	variables
╛trainable_variables
┐regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
╦
╪trace_0
┘trace_12Р
+__inference_dropout_12_layer_call_fn_534232
+__inference_dropout_12_layer_call_fn_534237│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0z┘trace_1
Б
┌trace_0
█trace_12╞
F__inference_dropout_12_layer_call_and_return_conditional_losses_534242
F__inference_dropout_12_layer_call_and_return_conditional_losses_534254│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0z█trace_1
"
_generic_user_object
0
╩0
╦1"
trackable_list_wrapper
0
╩0
╦1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
─	variables
┼trainable_variables
╞regularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
ю
сtrace_02╧
(__inference_dense_1_layer_call_fn_534263в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
Й
тtrace_02ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_534274в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0
": 
А┘2dense_1/kernel
:┘2dense_1/bias
 "
trackable_list_wrapper
▐
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40"
trackable_list_wrapper
0
у0
ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
°Bї
&__inference_model_layer_call_fn_530618input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
&__inference_model_layer_call_fn_532394inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
&__inference_model_layer_call_fn_532503inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
&__inference_model_layer_call_fn_531684input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
A__inference_model_layer_call_and_return_conditional_losses_532848inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
A__inference_model_layer_call_and_return_conditional_losses_533256inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
УBР
A__inference_model_layer_call_and_return_conditional_losses_531928input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
УBР
A__inference_model_layer_call_and_return_conditional_losses_532172input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠
┌0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
°20
∙21
·22
√23
№24
¤25
■26
 27
А28
Б29
В30
Г31
Д32
Е33
Ж34
З35
И36
Й37
К38
Л39
М40
Н41
О42
П43
Р44
С45
Т46
У47
Ф48
Х49
Ц50
Ч51
Ш52
Щ53
Ъ54
Ы55
Ь56
Э57
Ю58
Я59
а60
б61
в62
г63
д64
е65
ж66
з67
и68
й69
к70
л71
м72
н73
о74
п75
░76
▒77
▓78
│79
┤80
╡81
╢82
╖83
╕84
╣85
║86
╗87
╝88
╜89
╛90
┐91
└92
┴93
┬94
├95
─96
┼97
╞98
╟99
╚100
╔101
╩102
╦103
╠104"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
ъ
х0
ч1
щ2
ы3
э4
я5
ё6
є7
ї8
ў9
∙10
√11
¤12
 13
Б14
Г15
Е16
З17
Й18
Л19
Н20
П21
С22
У23
Х24
Ч25
Щ26
Ы27
Э28
Я29
б30
г31
е32
з33
й34
л35
н36
п37
▒38
│39
╡40
╖41
╣42
╗43
╜44
┐45
┴46
├47
┼48
╟49
╔50
╦51"
trackable_list_wrapper
ъ
ц0
ш1
ъ2
ь3
ю4
Ё5
Є6
Ї7
Ў8
°9
·10
№11
■12
А13
В14
Д15
Ж16
И17
К18
М19
О20
Р21
Т22
Ф23
Ц24
Ш25
Ъ26
Ь27
Ю28
а29
в30
д31
ж32
и33
к34
м35
о36
░37
▓38
┤39
╢40
╕41
║42
╝43
╛44
└45
┬46
─47
╞48
╚49
╩50
╠51"
trackable_list_wrapper
┐2╝╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
╦B╚
$__inference_signature_wrapper_532285input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
шBх
4__inference_layer_normalization_layer_call_fn_533265inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ГBА
O__inference_layer_normalization_layer_call_and_return_conditional_losses_533287inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
■B√
-__inference_kf_attention_layer_call_fn_533300inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
-__inference_kf_attention_layer_call_fn_533313inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
H__inference_kf_attention_layer_call_and_return_conditional_losses_533335inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
H__inference_kf_attention_layer_call_and_return_conditional_losses_533357inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
"
_generic_user_object
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
яBь
*__inference_dropout_1_layer_call_fn_533362inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
*__inference_dropout_1_layer_call_fn_533367inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_1_layer_call_and_return_conditional_losses_533372inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_1_layer_call_and_return_conditional_losses_533384inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_1_layer_call_fn_533393inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_533415inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
█B╪
'__inference_conv1d_layer_call_fn_533424inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_conv1d_layer_call_and_return_conditional_losses_533440inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
яBь
*__inference_dropout_2_layer_call_fn_533445inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
*__inference_dropout_2_layer_call_fn_533450inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_2_layer_call_and_return_conditional_losses_533455inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_2_layer_call_and_return_conditional_losses_533467inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_1_layer_call_fn_533476inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_1_layer_call_and_return_conditional_losses_533491inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_2_layer_call_fn_533500inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_533522inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
(
Б0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АB¤
/__inference_kf_attention_1_layer_call_fn_533535inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
/__inference_kf_attention_1_layer_call_fn_533548inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533570inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533592inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
"
_generic_user_object
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
яBь
*__inference_dropout_4_layer_call_fn_533597inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
*__inference_dropout_4_layer_call_fn_533602inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_4_layer_call_and_return_conditional_losses_533607inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_4_layer_call_and_return_conditional_losses_533619inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_3_layer_call_fn_533628inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_533650inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_2_layer_call_fn_533659inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_2_layer_call_and_return_conditional_losses_533675inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
яBь
*__inference_dropout_5_layer_call_fn_533680inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
*__inference_dropout_5_layer_call_fn_533685inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_5_layer_call_and_return_conditional_losses_533690inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_5_layer_call_and_return_conditional_losses_533702inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_3_layer_call_fn_533711inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_3_layer_call_and_return_conditional_losses_533726inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_4_layer_call_fn_533735inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_533757inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
(
└0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АB¤
/__inference_kf_attention_2_layer_call_fn_533770inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
/__inference_kf_attention_2_layer_call_fn_533783inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533805inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533827inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
"
_generic_user_object
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
яBь
*__inference_dropout_7_layer_call_fn_533832inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
*__inference_dropout_7_layer_call_fn_533837inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_7_layer_call_and_return_conditional_losses_533842inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_7_layer_call_and_return_conditional_losses_533854inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_5_layer_call_fn_533863inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_533885inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_4_layer_call_fn_533894inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_4_layer_call_and_return_conditional_losses_533910inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
яBь
*__inference_dropout_8_layer_call_fn_533915inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
*__inference_dropout_8_layer_call_fn_533920inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_8_layer_call_and_return_conditional_losses_533925inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
E__inference_dropout_8_layer_call_and_return_conditional_losses_533937inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_5_layer_call_fn_533946inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_5_layer_call_and_return_conditional_losses_533961inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_6_layer_call_fn_533970inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_533992inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
(
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АB¤
/__inference_kf_attention_3_layer_call_fn_534005inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
/__inference_kf_attention_3_layer_call_fn_534018inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534040inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534062inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
"
_generic_user_object
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
ЁBэ
+__inference_dropout_10_layer_call_fn_534067inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
+__inference_dropout_10_layer_call_fn_534072inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_10_layer_call_and_return_conditional_losses_534077inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_10_layer_call_and_return_conditional_losses_534089inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ъBч
6__inference_layer_normalization_7_layer_call_fn_534098inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_534120inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_6_layer_call_fn_534129inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_6_layer_call_and_return_conditional_losses_534145inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ЁBэ
+__inference_dropout_11_layer_call_fn_534150inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
+__inference_dropout_11_layer_call_fn_534155inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_11_layer_call_and_return_conditional_losses_534160inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_11_layer_call_and_return_conditional_losses_534172inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▌B┌
)__inference_conv1d_7_layer_call_fn_534181inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_conv1d_7_layer_call_and_return_conditional_losses_534196inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
·Bў
9__inference_global_average_pooling1d_layer_call_fn_534201inputs"п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_534207inputs"п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
┌B╫
&__inference_dense_layer_call_fn_534216inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense_layer_call_and_return_conditional_losses_534227inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ЁBэ
+__inference_dropout_12_layer_call_fn_534232inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
+__inference_dropout_12_layer_call_fn_534237inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_12_layer_call_and_return_conditional_losses_534242inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_12_layer_call_and_return_conditional_losses_534254inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▄B┘
(__inference_dense_1_layer_call_fn_534263inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_dense_1_layer_call_and_return_conditional_losses_534274inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
═	variables
╬	keras_api

╧total

╨count"
_tf_keras_metric
c
╤	variables
╥	keras_api

╙total

╘count
╒
_fn_kwargs"
_tf_keras_metric
,:*2 Adam/m/layer_normalization/gamma
,:*2 Adam/v/layer_normalization/gamma
+:)2Adam/m/layer_normalization/beta
+:)2Adam/v/layer_normalization/beta
5:3А2 Adam/m/kf_attention/query_kernel
5:3А2 Adam/v/kf_attention/query_kernel
5:3А2 Adam/m/kf_attention/value_kernel
5:3А2 Adam/v/kf_attention/value_kernel
::8А2%Adam/m/kf_attention/projection_kernel
::8А2%Adam/v/kf_attention/projection_kernel
/:-2#Adam/m/kf_attention/projection_bias
/:-2#Adam/v/kf_attention/projection_bias
.:,2"Adam/m/layer_normalization_1/gamma
.:,2"Adam/v/layer_normalization_1/gamma
-:+2!Adam/m/layer_normalization_1/beta
-:+2!Adam/v/layer_normalization_1/beta
(:&2Adam/m/conv1d/kernel
(:&2Adam/v/conv1d/kernel
:2Adam/m/conv1d/bias
:2Adam/v/conv1d/bias
*:(2Adam/m/conv1d_1/kernel
*:(2Adam/v/conv1d_1/kernel
 :2Adam/m/conv1d_1/bias
 :2Adam/v/conv1d_1/bias
.:,2"Adam/m/layer_normalization_2/gamma
.:,2"Adam/v/layer_normalization_2/gamma
-:+2!Adam/m/layer_normalization_2/beta
-:+2!Adam/v/layer_normalization_2/beta
7:5А2"Adam/m/kf_attention_1/query_kernel
7:5А2"Adam/v/kf_attention_1/query_kernel
7:5А2"Adam/m/kf_attention_1/value_kernel
7:5А2"Adam/v/kf_attention_1/value_kernel
<::А2'Adam/m/kf_attention_1/projection_kernel
<::А2'Adam/v/kf_attention_1/projection_kernel
1:/2%Adam/m/kf_attention_1/projection_bias
1:/2%Adam/v/kf_attention_1/projection_bias
.:,2"Adam/m/layer_normalization_3/gamma
.:,2"Adam/v/layer_normalization_3/gamma
-:+2!Adam/m/layer_normalization_3/beta
-:+2!Adam/v/layer_normalization_3/beta
*:(2Adam/m/conv1d_2/kernel
*:(2Adam/v/conv1d_2/kernel
 :2Adam/m/conv1d_2/bias
 :2Adam/v/conv1d_2/bias
*:(2Adam/m/conv1d_3/kernel
*:(2Adam/v/conv1d_3/kernel
 :2Adam/m/conv1d_3/bias
 :2Adam/v/conv1d_3/bias
.:,2"Adam/m/layer_normalization_4/gamma
.:,2"Adam/v/layer_normalization_4/gamma
-:+2!Adam/m/layer_normalization_4/beta
-:+2!Adam/v/layer_normalization_4/beta
7:5А2"Adam/m/kf_attention_2/query_kernel
7:5А2"Adam/v/kf_attention_2/query_kernel
7:5А2"Adam/m/kf_attention_2/value_kernel
7:5А2"Adam/v/kf_attention_2/value_kernel
<::А2'Adam/m/kf_attention_2/projection_kernel
<::А2'Adam/v/kf_attention_2/projection_kernel
1:/2%Adam/m/kf_attention_2/projection_bias
1:/2%Adam/v/kf_attention_2/projection_bias
.:,2"Adam/m/layer_normalization_5/gamma
.:,2"Adam/v/layer_normalization_5/gamma
-:+2!Adam/m/layer_normalization_5/beta
-:+2!Adam/v/layer_normalization_5/beta
*:(2Adam/m/conv1d_4/kernel
*:(2Adam/v/conv1d_4/kernel
 :2Adam/m/conv1d_4/bias
 :2Adam/v/conv1d_4/bias
*:(2Adam/m/conv1d_5/kernel
*:(2Adam/v/conv1d_5/kernel
 :2Adam/m/conv1d_5/bias
 :2Adam/v/conv1d_5/bias
.:,2"Adam/m/layer_normalization_6/gamma
.:,2"Adam/v/layer_normalization_6/gamma
-:+2!Adam/m/layer_normalization_6/beta
-:+2!Adam/v/layer_normalization_6/beta
7:5А2"Adam/m/kf_attention_3/query_kernel
7:5А2"Adam/v/kf_attention_3/query_kernel
7:5А2"Adam/m/kf_attention_3/value_kernel
7:5А2"Adam/v/kf_attention_3/value_kernel
<::А2'Adam/m/kf_attention_3/projection_kernel
<::А2'Adam/v/kf_attention_3/projection_kernel
1:/2%Adam/m/kf_attention_3/projection_bias
1:/2%Adam/v/kf_attention_3/projection_bias
.:,2"Adam/m/layer_normalization_7/gamma
.:,2"Adam/v/layer_normalization_7/gamma
-:+2!Adam/m/layer_normalization_7/beta
-:+2!Adam/v/layer_normalization_7/beta
*:(2Adam/m/conv1d_6/kernel
*:(2Adam/v/conv1d_6/kernel
 :2Adam/m/conv1d_6/bias
 :2Adam/v/conv1d_6/bias
*:(2Adam/m/conv1d_7/kernel
*:(2Adam/v/conv1d_7/kernel
 :2Adam/m/conv1d_7/bias
 :2Adam/v/conv1d_7/bias
$:"	
А2Adam/m/dense/kernel
$:"	
А2Adam/v/dense/kernel
:А2Adam/m/dense/bias
:А2Adam/v/dense/bias
':%
А┘2Adam/m/dense_1/kernel
':%
А┘2Adam/v/dense_1/kernel
 :┘2Adam/m/dense_1/bias
 :┘2Adam/v/dense_1/bias
0
╧0
╨1"
trackable_list_wrapper
.
═	variables"
_generic_user_object
:  (2total
:  (2count
0
╙0
╘1"
trackable_list_wrapper
.
╤	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperь
!__inference__wrapped_model_529676╞Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦4в1
*в'
%К"
input_1         

к "2к/
-
dense_1"К
dense_1         ┘│
D__inference_conv1d_1_layer_call_and_return_conditional_losses_533491kno3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Н
)__inference_conv1d_1_layer_call_fn_533476`no3в0
)в&
$К!
inputs         

к "%К"
unknown         
╡
D__inference_conv1d_2_layer_call_and_return_conditional_losses_533675mЭЮ3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ П
)__inference_conv1d_2_layer_call_fn_533659bЭЮ3в0
)в&
$К!
inputs         

к "%К"
unknown         
╡
D__inference_conv1d_3_layer_call_and_return_conditional_losses_533726mно3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ П
)__inference_conv1d_3_layer_call_fn_533711bно3в0
)в&
$К!
inputs         

к "%К"
unknown         
╡
D__inference_conv1d_4_layer_call_and_return_conditional_losses_533910m▄▌3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ П
)__inference_conv1d_4_layer_call_fn_533894b▄▌3в0
)в&
$К!
inputs         

к "%К"
unknown         
╡
D__inference_conv1d_5_layer_call_and_return_conditional_losses_533961mьэ3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ П
)__inference_conv1d_5_layer_call_fn_533946bьэ3в0
)в&
$К!
inputs         

к "%К"
unknown         
╡
D__inference_conv1d_6_layer_call_and_return_conditional_losses_534145mЫЬ3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ П
)__inference_conv1d_6_layer_call_fn_534129bЫЬ3в0
)в&
$К!
inputs         

к "%К"
unknown         
╡
D__inference_conv1d_7_layer_call_and_return_conditional_losses_534196mлм3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ П
)__inference_conv1d_7_layer_call_fn_534181bлм3в0
)в&
$К!
inputs         

к "%К"
unknown         
▒
B__inference_conv1d_layer_call_and_return_conditional_losses_533440k^_3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Л
'__inference_conv1d_layer_call_fn_533424`^_3в0
)в&
$К!
inputs         

к "%К"
unknown         
о
C__inference_dense_1_layer_call_and_return_conditional_losses_534274g╩╦0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         ┘
Ъ И
(__inference_dense_1_layer_call_fn_534263\╩╦0в-
&в#
!К
inputs         А
к ""К
unknown         ┘л
A__inference_dense_layer_call_and_return_conditional_losses_534227f╗╝/в,
%в"
 К
inputs         

к "-в*
#К 
tensor_0         А
Ъ Е
&__inference_dense_layer_call_fn_534216[╗╝/в,
%в"
 К
inputs         

к ""К
unknown         А╡
F__inference_dropout_10_layer_call_and_return_conditional_losses_534077k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ╡
F__inference_dropout_10_layer_call_and_return_conditional_losses_534089k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ П
+__inference_dropout_10_layer_call_fn_534067`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
П
+__inference_dropout_10_layer_call_fn_534072`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
╡
F__inference_dropout_11_layer_call_and_return_conditional_losses_534160k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ╡
F__inference_dropout_11_layer_call_and_return_conditional_losses_534172k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ П
+__inference_dropout_11_layer_call_fn_534150`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
П
+__inference_dropout_11_layer_call_fn_534155`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
п
F__inference_dropout_12_layer_call_and_return_conditional_losses_534242e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ п
F__inference_dropout_12_layer_call_and_return_conditional_losses_534254e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ Й
+__inference_dropout_12_layer_call_fn_534232Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АЙ
+__inference_dropout_12_layer_call_fn_534237Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         А┤
E__inference_dropout_1_layer_call_and_return_conditional_losses_533372k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ┤
E__inference_dropout_1_layer_call_and_return_conditional_losses_533384k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ О
*__inference_dropout_1_layer_call_fn_533362`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
О
*__inference_dropout_1_layer_call_fn_533367`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
┤
E__inference_dropout_2_layer_call_and_return_conditional_losses_533455k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ┤
E__inference_dropout_2_layer_call_and_return_conditional_losses_533467k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ О
*__inference_dropout_2_layer_call_fn_533445`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
О
*__inference_dropout_2_layer_call_fn_533450`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
┤
E__inference_dropout_4_layer_call_and_return_conditional_losses_533607k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ┤
E__inference_dropout_4_layer_call_and_return_conditional_losses_533619k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ О
*__inference_dropout_4_layer_call_fn_533597`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
О
*__inference_dropout_4_layer_call_fn_533602`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
┤
E__inference_dropout_5_layer_call_and_return_conditional_losses_533690k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ┤
E__inference_dropout_5_layer_call_and_return_conditional_losses_533702k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ О
*__inference_dropout_5_layer_call_fn_533680`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
О
*__inference_dropout_5_layer_call_fn_533685`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
┤
E__inference_dropout_7_layer_call_and_return_conditional_losses_533842k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ┤
E__inference_dropout_7_layer_call_and_return_conditional_losses_533854k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ О
*__inference_dropout_7_layer_call_fn_533832`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
О
*__inference_dropout_7_layer_call_fn_533837`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
┤
E__inference_dropout_8_layer_call_and_return_conditional_losses_533925k7в4
-в*
$К!
inputs         

p 
к "0в-
&К#
tensor_0         

Ъ ┤
E__inference_dropout_8_layer_call_and_return_conditional_losses_533937k7в4
-в*
$К!
inputs         

p
к "0в-
&К#
tensor_0         

Ъ О
*__inference_dropout_8_layer_call_fn_533915`7в4
-в*
$К!
inputs         

p 
к "%К"
unknown         
О
*__inference_dropout_8_layer_call_fn_533920`7в4
-в*
$К!
inputs         

p
к "%К"
unknown         
█
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_534207ВIвF
?в<
6К3
inputs'                           

 
к "5в2
+К(
tensor_0                  
Ъ ┤
9__inference_global_average_pooling1d_layer_call_fn_534201wIвF
?в<
6К3
inputs'                           

 
к "*К'
unknown                  ╟
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533570yВГДЕ;в8
1в.
$К!
inputs         

p

 
к "0в-
&К#
tensor_0         

Ъ ╟
J__inference_kf_attention_1_layer_call_and_return_conditional_losses_533592yВГДЕ;в8
1в.
$К!
inputs         

p 

 
к "0в-
&К#
tensor_0         

Ъ б
/__inference_kf_attention_1_layer_call_fn_533535nВГДЕ;в8
1в.
$К!
inputs         

p 

 
к "%К"
unknown         
б
/__inference_kf_attention_1_layer_call_fn_533548nВГДЕ;в8
1в.
$К!
inputs         

p

 
к "%К"
unknown         
╟
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533805y┴┬├─;в8
1в.
$К!
inputs         

p

 
к "0в-
&К#
tensor_0         

Ъ ╟
J__inference_kf_attention_2_layer_call_and_return_conditional_losses_533827y┴┬├─;в8
1в.
$К!
inputs         

p 

 
к "0в-
&К#
tensor_0         

Ъ б
/__inference_kf_attention_2_layer_call_fn_533770n┴┬├─;в8
1в.
$К!
inputs         

p 

 
к "%К"
unknown         
б
/__inference_kf_attention_2_layer_call_fn_533783n┴┬├─;в8
1в.
$К!
inputs         

p

 
к "%К"
unknown         
╟
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534040yАБВГ;в8
1в.
$К!
inputs         

p

 
к "0в-
&К#
tensor_0         

Ъ ╟
J__inference_kf_attention_3_layer_call_and_return_conditional_losses_534062yАБВГ;в8
1в.
$К!
inputs         

p 

 
к "0в-
&К#
tensor_0         

Ъ б
/__inference_kf_attention_3_layer_call_fn_534005nАБВГ;в8
1в.
$К!
inputs         

p 

 
к "%К"
unknown         
б
/__inference_kf_attention_3_layer_call_fn_534018nАБВГ;в8
1в.
$К!
inputs         

p

 
к "%К"
unknown         
┴
H__inference_kf_attention_layer_call_and_return_conditional_losses_533335uCDEF;в8
1в.
$К!
inputs         

p

 
к "0в-
&К#
tensor_0         

Ъ ┴
H__inference_kf_attention_layer_call_and_return_conditional_losses_533357uCDEF;в8
1в.
$К!
inputs         

p 

 
к "0в-
&К#
tensor_0         

Ъ Ы
-__inference_kf_attention_layer_call_fn_533300jCDEF;в8
1в.
$К!
inputs         

p 

 
к "%К"
unknown         
Ы
-__inference_kf_attention_layer_call_fn_533313jCDEF;в8
1в.
$К!
inputs         

p

 
к "%К"
unknown         
└
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_533415kVW3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ъ
6__inference_layer_normalization_1_layer_call_fn_533393`VW3в0
)в&
$К!
inputs         

к "%К"
unknown         
└
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_533522kyz3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ъ
6__inference_layer_normalization_2_layer_call_fn_533500`yz3в0
)в&
$К!
inputs         

к "%К"
unknown         
┬
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_533650mХЦ3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ь
6__inference_layer_normalization_3_layer_call_fn_533628bХЦ3в0
)в&
$К!
inputs         

к "%К"
unknown         
┬
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_533757m╕╣3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ь
6__inference_layer_normalization_4_layer_call_fn_533735b╕╣3в0
)в&
$К!
inputs         

к "%К"
unknown         
┬
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_533885m╘╒3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ь
6__inference_layer_normalization_5_layer_call_fn_533863b╘╒3в0
)в&
$К!
inputs         

к "%К"
unknown         
┬
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_533992mў°3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ь
6__inference_layer_normalization_6_layer_call_fn_533970bў°3в0
)в&
$К!
inputs         

к "%К"
unknown         
┬
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_534120mУФ3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ь
6__inference_layer_normalization_7_layer_call_fn_534098bУФ3в0
)в&
$К!
inputs         

к "%К"
unknown         
╛
O__inference_layer_normalization_layer_call_and_return_conditional_losses_533287k:;3в0
)в&
$К!
inputs         

к "0в-
&К#
tensor_0         

Ъ Ш
4__inference_layer_normalization_layer_call_fn_533265`:;3в0
)в&
$К!
inputs         

к "%К"
unknown         
П
A__inference_model_layer_call_and_return_conditional_losses_531928╔Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦<в9
2в/
%К"
input_1         

p 

 
к "-в*
#К 
tensor_0         ┘
Ъ П
A__inference_model_layer_call_and_return_conditional_losses_532172╔Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦<в9
2в/
%К"
input_1         

p

 
к "-в*
#К 
tensor_0         ┘
Ъ О
A__inference_model_layer_call_and_return_conditional_losses_532848╚Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦;в8
1в.
$К!
inputs         

p 

 
к "-в*
#К 
tensor_0         ┘
Ъ О
A__inference_model_layer_call_and_return_conditional_losses_533256╚Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦;в8
1в.
$К!
inputs         

p

 
к "-в*
#К 
tensor_0         ┘
Ъ щ
&__inference_model_layer_call_fn_530618╛Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦<в9
2в/
%К"
input_1         

p 

 
к ""К
unknown         ┘щ
&__inference_model_layer_call_fn_531684╛Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦<в9
2в/
%К"
input_1         

p

 
к ""К
unknown         ┘ш
&__inference_model_layer_call_fn_532394╜Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦;в8
1в.
$К!
inputs         

p 

 
к ""К
unknown         ┘ш
&__inference_model_layer_call_fn_532503╜Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦;в8
1в.
$К!
inputs         

p

 
к ""К
unknown         ┘·
$__inference_signature_wrapper_532285╤Z:;CDEFVW^_noyzВГДЕХЦЭЮно╕╣┴┬├─╘╒▄▌ьэў°АБВГУФЫЬлм╗╝╩╦?в<
в 
5к2
0
input_1%К"
input_1         
"2к/
-
dense_1"К
dense_1         ┘