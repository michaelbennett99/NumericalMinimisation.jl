module NumericalMinimisation

using NumericalMethods, LinearAlgebra, ForwardDiff, DiffResults

export brent, newton

mutable struct ConvergenceError <: Exception
    msg::AbstractString
end

const cgold = 0.3819660

"""
This code, drawn from Numerical Recipes, implements Brent's method (without derivative) to
minimize the function f.  The arguments ax, bx, and cx must satisfy: ax < bx < cx, f(ax) > f(bx),
and f(cx) > f(bx).  The function returns (fmin,xmin,niter), where fmin is the minimized value of
the function f, xmin is the minimand, and niter is the number of iterations required to
find the minimum.  When using brent, good choices for the arguments tol and zeps are:
tol = zeps = 1.0e-8.  Here is an example of how to "call" brent:

fmin,xmin,niter = brent(ax,bx,cx,f,1.0e-8,1.0e-8).
"""
function brent(f, ax, bx, cx; atol=1e-6, rtol=1e-6, max_iter=1000)
    a = min(ax, cx)
    b = max(ax, cx)
    v = bx
    w = v
    x = v
    e = 0
    fx = f(x)
    fv = fx
    fw = fx
    
    d = 0
    
    for iter in 1:max_iter
        xm = 0.5*(a+b)
        tol1 = rtol*abs(x) + atol
        tol2 = 2.0*tol1
       
        if (abs(x-xm) <= (tol2-0.5*(b-a))) # Convergence check
            fmin = fx
            xmin = x
            niter = iter
            return xmin, fmin, niter
        end   
       
        if (abs(e) > tol1) 
            r = (x-w)*(fx-fv)
            q = (x-v)*(fx-fw)
            p = (x-v)*q - (x-w)*r
            q = 2.0*(q-r)
            
            if (q > 0) 
                p = -p
            end   
            
            q = abs(q)
            etemp = e
            e = d
            
            if ( (abs(p) >= abs(0.5*q*etemp)) |
                (p <= (q*(a-x))) | (p >= (q*(b-x))) ) 
                if (x >= xm) 
                e = a - x
                else
                e = b - x
                end
                d = cgold*e
            else   
                d = p/q
                u = x + d
                if ( ((u-a) < tol2) | ((b-u) < tol2) )
                d = abs(tol1)*sign(xm-x)
                end   
            end  
        else
            if (x >= xm) 
                e = a - x
            else
                e = b - x
            end      
            d = cgold*e
        end
        if (abs(d) >= tol1) 
            u = x + d
        else
            u = x + abs(tol1)*sign(d)
        end
        fu = f(u)
        if (fu <= fx) 
            if (u >= x) 
                a = x
            else
                b = x
            end
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else
            if (u < x) 
                a = u
            else
                b = u
            end
            if ( (fu <= fw) | (abs(w-x) <= atol) )
                v = w
                fv = fw
                w = u
                fw = fu
            elseif ( (fu <= fv) | (abs(v-x) <= atol) | 
                    (abs(v-w) <= atol) ) 
                v = u
                fv = fu
            end
        end 
    end

    throw(ConvergenceError("Maximum number of iterations exceeded in brent."))
end

function newton(
        f::Function, x_0::Real, δ; atol=1e-6, rtol=1e-6, max_iter=1000
    )
    x_1 = x_0
    for i in 1:max_iter
        x_2 = x_1 - numderiv_two_side(f, x_1, δ=δ)/numderiv_second(f, x_1, δ=δ)
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end

function newton(
        f::Function, x_0::AbstractVector{<:Real}, δ;
        atol=1e-6, rtol=1e-6, max_iter=1000
    )
    x_1 = x_0
    for i in 1:max_iter
        hess = Symmetric(hessian(f, x_1, δ=δ))
        grad = gradient(f, x_1, δ=δ)
        Δx = hess\grad
        x_2 = x_1 - Δx
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end

function newton(
    f::Function, x_0::Real; atol=1e-6, rtol=1e-6, max_iter=1000
    )
    dx(z) = ForwardDiff.derivative(f, z)
    x_1 = x_0
    for i in 1:max_iter
        dx = dx(x_1)
        d2x = ForwardDiff.derivative(dx, x_1)
        x_2 = x_1 - d2x\dx
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end

function newton(
    f::Function, x_0::AbstractVector{<:Real};
    atol=1e-6, rtol=1e-6, max_iter=1000
)
x_1 = x_0
for i in 1:max_iter
    result = DiffResults.HessianResult(x_1)
    result = ForwardDiff.hessian!(result, f, x_1)
    hess = Symmetric(DiffResults.hessian(result))
    grad = DiffResults.gradient(result)
    Δx = hess\grad
    x_2 = x_1 - Δx
    if isapprox(x_2, x_1, atol=atol, rtol=rtol)
        return x_2, f(x_2), i
    end
    x_1 = x_2
end
throw(ConvergenceError("Maximum number of iterations exceeded."))
end

end # module
