%%%%%%%%%%% INITIAL VALUES AND SETUP %%%%%%%%%%%%

%N number of Data Points
N = 100;

%create N random data points in R^3
x = rand([N 3]);
%create e       %e = 0 for part a-c
e = .1*rand([N 1]);
%create y
y = x(:,1).*x(:,2) + x(:,3) + e(:);
% y = x(:,1).*x(:,3) + 1.5*x(:,1);     Different Function
%lambda and it's matrix
lambda = 10^-5; 
lambdaSquaredRootedMatrix = sqrt(lambda) * eye(16);
%initial weights
w = ones([16 1]);
%gamma initialization
gamma = 10^-5;
%tanh equation setup
syms n;
tanh = @(n) ((exp(n)-exp(-n))/(exp(n) + exp(-n)));                              
%setup given equation for f_w
syms x1 x2 x3 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16;
f = @(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) ...
    w1*tanh(w2*x1+w3*x2+w4*x3+w5) + w6*tanh(w7*x1+w8*x2+w9*x3+w10) + ...
    w11*tanh(w12*x1+w13*x2+w14*x3+w15) + w16;

%%%%%%%%%%% END OF INITIAL SETUP %%%%%%%%%%%%%%%%

for i = 1:300   %imax is 300

    initialDerivativeMatrix = derivativeMatrix(x,w,N);
    
    %residual
    r = zeros([N 1]);
    for k = 1:N
        r(k) = f(x(k,1),x(k,2),x(k,3),w(1),w(2),w(3),w(4),w(5),w(6),...
            w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)) - y(k,1);
    end
             
    
    %b_t
    b = r - initialDerivativeMatrix * w;
    %append zeros
    b = [b;zeros([16 1])];
    %D_h 
    Dh = [initialDerivativeMatrix; lambdaSquaredRootedMatrix];
    


    %final b for iteration
    b = [b; -1*sqrt(gamma)*w];
    %final 'A' [b - (-A)w]
    A = [Dh; (sqrt(gamma) * eye(16))];
    A = -1*A;
    %new w
    new_weights = pinv(transpose(A) * A) * transpose(A) * b;
    


    %residual of k+1
    r_hat = zeros([N 1]);
    for u = 1:N
    r_hat(u) = f(x(u,1),x(u,2),x(u,3),new_weights(1),new_weights(2),...
        new_weights(3),new_weights(4),new_weights(5),new_weights(6),...
        new_weights(7),new_weights(8),new_weights(9),new_weights(10),...
        new_weights(11),new_weights(12),new_weights(13),new_weights(14),...
        new_weights(15),new_weights(16)) - y(u,1);
    end



    %loss function for k+1
    h_hat = [r_hat; (lambda*new_weights)];
    loss_hat(i) = transpose(h_hat)*h_hat;
    
    %loss function for k
    h_bar = [r; (lambda*w)];
    loss_bar(i) = transpose(h_bar)*h_bar;
    
    %loss function comparisons
    if (loss_hat(i) < loss_bar(i)) %loss function of loss hat is better, so update weights  
        w = new_weights;
        gamma = .8*gamma;
    else %increase gamma and redo LS
        gamma = 2*gamma;
    end



    %finishing is when the loss functions are withing 10^-4 
    %or less than 10^-3
    if(abs(loss_hat(i) - loss_bar(i)) < 10^-7)
        break;
    end
    
    %or when loss functions are small
    if loss_hat(i) < 10^-2
        w = new_weights;
        break;
    elseif loss_bar(i) < 10^-2
        break;
    end
   

end





%returns derivative matrix of all N Points (x Points for each row, weights
%in each column
function Dr = derivativeMatrix(x,w,N)

    %setup given equation for f_w
    syms x1 x2 x3 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16;
    f = @(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) ...
        w1*tanh(w2*x1+w3*x2+w4*x3+w5) + w6*tanh(w7*x1+w8*x2+w9*x3+w10) + ...
        w11*tanh(w12*x1+w13*x2+w14*x3+w15) + w16;

    %differential w/r to each weight
    Dr1(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w1);
    Dr2(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w2);
    Dr3(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w3);
    Dr4(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w4);
    Dr5(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w5);
    Dr6(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w6);
    Dr7(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w7);
    Dr8(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w8);
    Dr9(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w9);
    Dr10(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w10);
    Dr11(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w11);
    Dr12(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w12);
    Dr13(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w13);
    Dr14(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w14);
    Dr15(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w15);
    Dr16(x1,x2,x3,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16) = diff(f,w16);
    
    %skeleton derivative matrix for all N points (500)
    Dr = zeros([N 16]);
  
    %evaluate each partial derivative (w1,w2,...,w16 - columns;
    %x[(1,1),x(1,2),x(1,3)] ... x[(N,1),x(N,2),x(N,3)] - N rows (500)
    Dr = double([Dr1(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),...
                 Dr2(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),...
                 Dr3(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr4(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr5(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr6(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr7(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr8(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr9(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr10(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr11(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr12(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr13(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr14(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr15(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16)),... 
                 Dr16(x(:,1),x(:,2),x(:,3),w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10),w(11),w(12),w(13),w(14),w(15),w(16))]);                                                                       
end




