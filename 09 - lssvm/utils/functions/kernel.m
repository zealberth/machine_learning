function k = kernel(params,u,v)
    if params.kernel == 1       % linear
        k = u * v';
    elseif params.kernel == 2   % rbf / gaussian
        k = exp(-(1/(2*params.sigma^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
                -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
    elseif params.kernel == 3   % quadratic
        dotproduct = (u*v');
        k = dotproduct.*(1 + dotproduct);
    end
end