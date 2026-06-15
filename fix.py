#!/usr/bin/env python
import sys

try:
    with open('harness/profiles.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # replace any weird tabs/spaces inside the function definition
    lines = content.split('\n')
    for i in range(96, 105):
        if i < len(lines):
            # strip and manually indent with 4 spaces for the docstring
            if 'Resolve ONE' in lines[i]:
                lines[i] = '    Resolve ONE profile for an agent process.'
            elif 'Priority:' in lines[i]:
                lines[i] = '    Priority: explicit `name` arg -> AGENT_PROFILE env var -> \'anchor\'.'
            elif 'This is what' in lines[i]:
                lines[i] = '    This is what `agent.py --profile` and the arena deployments use; four arena'
            elif 'identities =' in lines[i]:
                lines[i] = '    identities = four processes, each with its own AGENT_PROFILE + API key.'
            elif lines[i].strip() == '\"\"\"' and i > 97 and i < 105:
                lines[i] = '    \"\"\"'
    
    with open('harness/profiles.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Fixed harness/profiles.py')
except Exception as e:
    print('Error:', e)
