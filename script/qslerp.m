function qt = qslerp(q1, q2, t)
% Quaternion slerp interpolation as defined in transformations.py
% INPUT
%  q1 q2 quaternions
%  t interpolation midpoint within range [0,1[
% OUTPUT
%  qt interpolated quaternion
    
if t==0.0
    qt = q1;
    return
elseif t==1.0
    qt = q2;
    return
end

% Invert the quaternion if needed to assure the shortest path gets taken by
% the interpolation
d = q1*q2';
if d<0.0
    d = -d;
    q1 = q1 * -1.0;
end

angle = acos(d);
if angle==0
    qt = q1;
    return
end

isin = 1.0 / sin(angle);
t1 = q1 * sin((1.0-t)*angle)*isin;
t2 = q2 * sin(t*angle)*isin;
qt = t1+t2;
    
end