% This function generates accerlation data from position and velocity
% where the system is assumed to be the mass-spring-damper system.
% mass = m, position of the mass = p, velocity of the mass = v. 
% recovering force = - (k1 p + k2 p^3), damping force = -(c1 v + c2 v^3)

function [position, velocity, acceleration] = dataGeneration(k1,k2,c1,c2,m,pGrid,vGrid)
 
pmin = -1; pmax = 1; vmin = -1; vmax = 1;

position = pmin:pGrid:pmax;
velocity = vmin:vGrid:vmax;

acceleration = zeros(length(position),length(velocity));

for ii = 1:length(position)
    for jj = 1:length(velocity)
        p = position(ii); 
        v = velocity(jj);
        acceleration(ii,jj) = -(k1*p + c1*v + k2*p^3 + c2*v^3)/m;
    end
end

end