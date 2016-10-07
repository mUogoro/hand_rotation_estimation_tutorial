% hand_utils.m
% The hand_utils class provides some utility functions for animating a
% realistic (30DOF) hand skeletal model.
% Note: most of the functions work on a pose vector. More details about the
% pose vector can be found in hand_poses.m

classdef hand_utils
   properties (Constant) 

        % Skeleton bone names
        % Note: the vector above define also the index associated to each
        % bone in the following vectors
        HAND_BONE_NAME = { ...
            'hand_R', ... % Approcimate metacarpal bones (palm area)
            'wrist_R', ... % Carpal bones (wrist area)
            'palm_pinky_R', 'pinky_1_R', 'pinky_2_R', 'pinky_3_R', ... % Pinky CMC, MCP, PIP and DIP
            'palm_ring_R', 'ring_1_R', 'ring_2_R', 'ring_3_R', ... % Ring CMC, MCP, PIP and DIP
            'palm_middle_R', 'middle_1_R', 'middle_2_R', 'middle_3_R', ... % Middle CMC, MCP, PIP and DIP
            'palm_index_R', 'index_1_R', 'index_2_R', 'index_3_R', ... % Index CMC, MCP, PIP and DIP
            'thumb_1_R', 'thumb_2_R', 'thumb_3_R'}; % Thumb CMC, MCP and DIP
        HAND_NBONES = 21;
        
        % Axis of rotation by which each bone can be rotated
        % Note: except for the first two bones, the rotations about the
        % axis define:
        % - X flexion/extension
        % - Y torsion
        % - Z adduction/abduction
        HAND_BONE_RAXIS = {'XYZ', ...
                   'XZ', ...
                   'X', 'ZX', 'X', 'X', ...
                   'X', 'ZX', 'X', 'X', ...
                   'X', 'ZX', 'X', 'X', ...
                   'X', 'ZX', 'X', 'X', ...
                   'ZX', 'ZX', 'X'};
        HAND_POSE_LEN = 30;
        
        % Upper and lower bounds for the values of the pose vector
        % Note: values are in degrees
        HAND_BONE_RLOWBOUND = {[-180, -180, -180], ...
                        [-30, -60], ...
                        [0], [-40, -15], [0], [-10], ...
                        [0], [-40, -15], [0], [-10], ...
                        [0], [-30, -15], [0], [-10], ...
                        [0], [-30, -15], [0], [-10], ...
                        [-45, -45], [-45, -45], [-15]};
        HAND_BONE_RUPBOUND = {[180, 180, 180], ...
                      [30, 90]...
                      [30], [40, 90], [110], [90], ...
                      [30], [40, 90], [110], [90], ...
                      [5], [30, 90], [110], [90], ...
                      [5], [30, 90], [110], [90], ...
                      [45, 45], [45, 45], [90]};

   end
   
   methods (Static)
       function [ppose] = perturb_pose(pose, sigma)
       % Perturb a pose by adding a random (normal) noise to each pose entry.
       % INPUT
       %  pose the pose vector
       %  sigma noise std (in degrees)
       % OUTPUT
       %  ppose the perturbed pose

          ppose = pose(:)';
          startIdx = 1;
          for i=1:hand_utils.HAND_NBONES
              endIdx = startIdx+length(hand_utils.HAND_BONE_RAXIS{i})-1;
              ppose(startIdx:endIdx) = normrnd(pose(startIdx:endIdx), ...
                                               sigma(startIdx:endIdx));
              % Limit the pose angles within allowed range
              ppose(startIdx:endIdx) = ...
                  max(min(ppose(startIdx:endIdx), ...
                          hand_utils.HAND_BONE_RUPBOUND{i}), ...
                      hand_utils.HAND_BONE_RLOWBOUND{i});
              startIdx = endIdx+1;
          end
       end
       
       function [pq] = perturb_rotation(q, sigma)
       % Add a random perturbation to a rotation
       % INPUT
       %  q input rotation expressed as a quaternion
       %  sigma std of the normal perturbation
       % OUTPUT
       %  pq the perturbed quaternion
           ax = [rand() rand() rand() 0.];
           ax = ax/norm(ax);
           ax(4) = normrnd(0, deg2rad(sigma));
           
           pq = rotm2quat(axang2rotm(ax)*quat2rotm(q));
       end
       
       function [rotm] = pose2rotm(pose)
       % Convert a pose vector into a vector or per-bone rotation matrices
       % INPUT
       %  pose the pose vector
       % OUTPUT
       %  rotm a 4x4xN matrix, where N is the number of hand bones
           dof=1;
           rotm = zeros(4, 4, hand_utils.HAND_NBONES);
           for i=1:hand_utils.HAND_NBONES
              rotm(:,:,i) = eye(4);
              raxis = hand_utils.HAND_BONE_RAXIS{i};
              for j=1:length(raxis)
                  ax = [raxis(j)=='X' raxis(j)=='Y' raxis(j)=='Z' ...
                        deg2rad(pose(dof))];
                  rotm(1:3,1:3,i) = axang2rotm(ax)*rotm(1:3,1:3,i);
                  dof = dof+1;
              end
           end
       end
       
       function [ipose] = interpPoses(pose1, pose2, t)
       % Interpolate between two poses
       % INPUT
       %  pose1 start pose
       %  pose2 end pose
       %  t interpolation position within the range [0,1[
       % OUTPUT
       %  ipose the interpolated pose
           ipose = zeros(1, length(pose1));
           startIdx = 1;
           for i=1:hand_utils.HAND_NBONES
               raxis = hand_utils.HAND_BONE_RAXIS{i};
               endIdx = startIdx+length(raxis)-1;
               
               bone1q = [1 0 0 0];
               bone2q = [1 0 0 0];
               for j=1:length(raxis)
                  
                  switch raxis(j)
                      case 'X', ax = [1 0 0];
                      case 'Y', ax = [0 1 0];
                      case 'Z', ax = [0 0 1];
                  end
                  
                  boneax1 = cat(2, ax, deg2rad(pose1(startIdx+j-1)));
                  boneax2 = cat(2, ax, deg2rad(pose2(startIdx+j-1)));
                  bone1q = quatmultiply(axang2quat(boneax1), bone1q);
                  bone2q = quatmultiply(axang2quat(boneax2), bone2q);
               end
               
               iboneq = qslerp(bone1q, bone2q, t);
               
               % TODO: handle generic pose here
               if strcmp(raxis, 'XYZ'),    [rz, ry, rx] = quat2angle(iboneq, 'ZYX'); ipose(startIdx:endIdx) = rad2deg([rx ry rz]);
               elseif strcmp(raxis, 'XZ'), [~, rz, rx]  = quat2angle(iboneq, 'YZX'); ipose(startIdx:endIdx) = rad2deg([rx rz]);
               elseif strcmp(raxis, 'ZX'), [~, rx, rz]  = quat2angle(iboneq, 'YXZ'); ipose(startIdx:endIdx) = rad2deg([rz rx]);
               elseif strcmp(raxis, 'X'),  [~, ~, rx]   = quat2angle(iboneq, 'ZYX'); ipose(startIdx:endIdx) = rad2deg(rx);
               end
               
               startIdx = endIdx+1;
           end
       end
   end
end