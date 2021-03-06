function [loss,grad] = stackedAutoencoderLoss(theta, netconfig, weight_decay, ...
                                             beta, noisedata, data, layerstype)
                                         
% stackedAutoencoderLoss: Takes a trained softmaxTheta and a training data 
%set with labels,and returns cost and gradient using a stacked autoencoder
% model. Used for finetuning.

% theta: trained weights from the autoencoder
% netconfig: the network configuration of the stack
% weight_decay: the weight regularization penalty
% noisedata: corrupted data 
% beta: sparity for activation ( not used yet)
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.

%% Unroll the stacked Autoencode parameter

% Extract out the "stack"
layersize=length(netconfig)-1;
stack=cell(layersize,1);
pos=0;
for ii=1:layersize
    stack{ii}=struct;
    wlen =  netconfig(ii+1)* netconfig(ii);
    stack{ii}.w = reshape( theta(pos+1:pos + wlen), netconfig(ii+1), netconfig(ii));
    pos=pos + wlen;
    blen = netconfig(ii+1);
    stack{ii}.b =  theta(pos+1:pos+netconfig(ii+1));
    pos=pos + blen;
end

% You will need to compute the following gradients
stackgrad = cell(size(stack));

m = size(data, 2);

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = noisedata;

for layer = (1:depth)
    %fprintf(' w = %d\n b = %d\n x = %d,y = %d\n', size(stack{layer}.w,2), size(stack{layer}.b,1), size(a{layer},1),size(a{layer},2));
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  if layerstype(layer+1)
    a{layer+1} = sigmoid(z{layer+1});
  else
    a{layer+1} = z{layer+1};
  end
end

squares = (a{depth+1} - data).^2;
squared_err_J = (1/2) * (1/m) * sum(squares(:));

weight_decay_J=0;
for layer=(1:depth)
    weight_decay_J = weight_decay_J + sum(stack{layer}.w(:).^2);
end
weight_decay_J = (weight_decay/depth) * weight_decay_J;

%sparsity_J = beta * KLsum;

loss = squared_err_J + weight_decay_J ;

% fprintf('softmaxTheta: '); disp(size(softmaxTheta));
% fprintf('softmaxThetaGrad: '); disp(size(softmaxThetaGrad));
% fprintf('a{depth+1}: '); disp(size(a{depth+1}));
% fprintf('groundTruth: '); disp(size(groundTruth));

if nargout >1
    
    d{depth+1} = -(data - a{depth+1}) ;
    if layerstype(depth+1)
     d{depth+1}= d{depth+1}.* a{depth+1} .* (1-a{depth+1});
    end
        

    for layer = (depth:-1:2)
      d{layer} = (stack{layer}.w' * d{layer+1});
      if layerstype(layer)
        d{layer} =d{layer}.* a{layer} .* (1-a{layer});
      end
    end

    for layer = (depth:-1:1)
      stackgrad{layer}.w = (1/m) * d{layer+1} * a{layer}'+ weight_decay * stack{layer}.w;
      stackgrad{layer}.b = (1/m) * sum(d{layer+1}, 2);
    end

    % -------------------------------------------------------------------------
    % ���ݶȷ�װ
    % -------------------------------------------------------------------------
     grad=[];
     for layer = (1:depth)
        grad = [grad; stackgrad{layer}.w(:); stackgrad{layer}.b(:)];
     end
     
end

end

% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
