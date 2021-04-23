"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain
import mods


episodeCount = 1000  # Number of learning episodes

sim = RoverDomainGym()
#mods.sequentialPoi(sim)
mods.recipePoi(sim)
#mods.lowVisibility(sim)
obs=sim.reset()

sim.data["Coupling"]=3

obs_size=len(obs[0])

print(obs_size)
ccea.initCcea(input_shape=obs_size, num_outputs=2, num_units=32)(sim.data)

for episodeIndex in range(episodeCount):
    sim.data["Episode Index"] = episodeIndex
    populationSize=len(sim.data['Agent Populations'][0])
    GlobalRewards=[0.0]
    
    for worldIndex in range(populationSize):
        sim.data["World Index"]=worldIndex
        
        obs = sim.reset()
         
        #ccea.assignCceaPolicies(sim.data)
        mods.assignHomogeneousPolicy(sim)

        done = False
        stepCount = 0
        
        while not done:

            #mods.poiVelocity(sim)
        
            # Select actions and create the joint action from the simulation data
            # Note that this specific function extracts "obs" from the data structure directly, which is why obs is not
            # directly used in this example.
            
            domain.doAgentProcess(sim.data)
            #mods.abilityVariation(sim)
            
            jointAction = sim.data["Agent Actions"]

            obs, reward, done, info = sim.step(jointAction)
            
            stepCount += 1
            if ( episodeIndex%50==49 and worldIndex==0):
                sim.render()
                
                
        GlobalRewards.append(sim.data["Global Reward"])    
        ccea.rewardCceaPolicies(sim.data)
        
    ccea.evolveCceaPolicies(sim.data)
    print(episodeIndex,max(GlobalRewards))
