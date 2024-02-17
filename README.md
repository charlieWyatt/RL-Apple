# To run -
python run_simulation.py



## Project Exploration Summary -

For my first weekly project, I have created my first Reinforcement Learning agent. While I have lots of experience across many ML domains, I’ve always been a far admirer of RL. But that changed this week!

I made a very simple agent which can see “apples” and changes its acceleration to try and collect as many as it can before the time expires.

Results- a great little agent that moves around in a very realistic way!



Challenges- 
1. learning the gymnasium and stable_baselines3 libraries. After some reading of documentation and (interrogation of chatgpt), they were pretty simple to pick up! If anyone is interested in RL highly recommend these libraries to understand the infrastructure of how to create an agent. 
2. Making the agent move in a natural way. Lots of fiddling with max velocities, max accelerations and reward functions until I got the agent moving smoothly. I also learnt the importance of “incremental rewards”. Trying to give the agent one big reward when it reaches the apple means during training it’s pretty unlikely to randomly hit it. But by penalising it by the distance away from the Apple every frame, it learns much more quickly that “close to apple”=good and will quickly move towards it. After awhile, I had to remove this distance penalty because the AI would often be so scared of “overshooting” the apple and spending time far away from the next one, that getting the original apple wasn’t worth it.
3. Getting stuck on the wall. For some reason, the AI loved the wall even after lots of training. I fixed this be penalising touching the wall. It it touches the wall, I give the AI a bad number. AI’s hate bad numbers. And so now, the AI hates walls.
