function [V,optpars,optllf] = estGARCH(R,pars0)
% estimates GARCH(1,1) model
%   R_t = sqrt(V_t)*Z_t (Returns are demeaned)
%   V_t = a + b*V_{t-1} + c*V_{t-1}*(Z_{t-1})^2
% pars=[a,b,c]
% fix mu = mean(R);
% V is unannualized

%pars0=[2*10^-6, .9, .09];

T=length(R);
R=R-mean(R);
V1=R(1)^2;
options=optimset('MaxFunEvals',15000,'MaxIter',10000,'TolFun',.001);
[optpars,optllf] = fminsearch(@llffun,pars0,options);
optllf=-T/2*log(2*pi)-.5*optllf;

function [LLF,V] = llffun(pars)
    a=pars(1); b=pars(2); c=pars(3); 
    try d=pars(4); catch, d=0; end
    V=zeros(T,1);
    V(1)=V1;
    LLF=0; Z=R(1)/sqrt(V(1));
     for t=2:T        
         V(t)=a+b*V(t-1)+c*V(t-1)*(Z-d)^2;
         Z=R(t)/sqrt(V(t));
         LLF = LLF + log(V(t)) + Z^2; 
     end
end

[~,V] = llffun(optpars);


end
     