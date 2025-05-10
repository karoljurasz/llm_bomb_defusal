
import asyncio
from agents.two_agents import run_two_agents
from agents.models import SmollLLM
import csv

defuser_model = SmollLLM('Qwen/Qwen2.5-0.5B-Instruct', device='cpu')
expert_model = SmollLLM('Qwen/Qwen2.5-0.5B-Instruct', device='cpu')

async def main():
    steps, modules = await run_two_agents(
        defuser_model=defuser_model,
        expert_model=expert_model,
        server_url='http://localhost:8115',
        max_new_tokens=150,
        top_p=0.6,
        top_k=50,
        temperature=0.9,
        max_steps=10,
        prompt_type='standard'
    )
    with open(r'results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['standard', 0.6, 50, 0.9, 3, steps, modules])

asyncio.run(main())
