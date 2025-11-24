import httpx
import asyncio
import uuid

import re
from typing import Dict

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Part,
    TextPart,
    MessageSendParams,
    Message,
    Role,
    SendMessageRequest,
    SendMessageResponse,
)


async def get_agent_card(url: str) -> AgentCard | None:
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)

    card: AgentCard | None = await resolver.get_agent_card()

    return card


async def wait_agent_ready(url, timeout=10):
    # wait until the A2A server is ready, check by getting the agent card
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                return True
            else:
                print(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def send_message(
    url, message, task_id=None, context_id=None
) -> SendMessageResponse:
    card = await get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=120.0)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)

    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    response = await client.send_message(request=req)
    return response

def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """the target str contains tags in the format of <tag_name> ... </tag_name>, parse them out and return a dict"""

    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}
