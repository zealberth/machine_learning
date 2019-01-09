function resultado = gaussiana(x, mi, sigma)
    a = exp(-0.5 * (x - mi) * pinv(sigma) * (x - mi)');
%     a = exp(-0.5 * ((x - mi) / sigma) * (x - mi)');
    ci =  (2*pi)^(size(x, 2)/2) * det(sigma)^(1/2);
    resultado = a/ci;
end