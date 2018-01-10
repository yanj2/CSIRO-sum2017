"""
Particle Swarm Optimisation improvements parabaloid - Jie Jenny Yan, January 2018
Using DEAP framework

Algorithm:

1) swarm initialisation
    - for each particle in the swarm, initialise the position from a uniform
      distribution with lower b and upper b (tbc)
    - find global best position while initialising the swarm
    - sample the velocity per particle from a uniform distribution

2) track prior global best (??)

3) swarm evolution
    - for each particle in the swarm:
        - sample the r_p, r_g values from uniform distribution(0,1)
        - with w = phi_p = phi_g = 0.5, calculate the new velocity with:

          v = w * v + phi_p * r_p * (best pos - curr pos) + phi_g * r_g * (glob best - curr pos)

        - update position
        - if fitness new position better than fitness of best position,
            - update best position
            - if best pos better than global best,
                - update global best
                - check <termination conditions>
    - update generation
    - update prev best

4) termination conditions
    - global best displacement smaller than delta
    - fitness value of best increased by less than threshold epsilon
    - exceeded max generations
    
5) return global best

"""
