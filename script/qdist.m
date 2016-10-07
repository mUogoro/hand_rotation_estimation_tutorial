function [d2] = qdist(XI, XJ)
  % Compute the quaternion distance, as defined in
  % R. Hartley, J. Trumpf, Y. Dai, and H. Li, "Rotation Averaging",
  % Int. J. Comput. Vis., vol. 103, no. 3, pp. 267-305, 2013.
  % INPUT
  %  XI XJ quaternions
  % OUPUT
  %  d2 distance (radians)
  
  nq = size(XJ,1);
  t = quatmultiply(repmat(quatinv(XI),nq,1) ,XJ);
  d2 = 2*acos(abs(t(:,1)));

end
