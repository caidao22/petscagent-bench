from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from src.util.a2a_comm import parse_tags
from .mcp_client import MCPClient
from .agent import Agent

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class GreenAgentExecutor(AgentExecutor):
    def __init__(self):
        self.client = MCPClient()
        self.agents: dict[str, Agent] = {}  # context_id to agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the context to get purple agent URL and other configurations
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        purple_agent_url = tags["purple_agent_url"]

        # create a new task
        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        context_id = task.context_id
        agent = Agent(purple_agent_url=purple_agent_url, mcp_client=self.client)
        self.agents[context_id] = agent
        agent = self.agents.get(context_id)
        updater = TaskUpdater(event_queue, task.id, context_id)

        print("Green agent: Start working...")
        await updater.start_work()
        try:
            await self.client.connect_to_local_server(
                "./tools/petsc_mcp_servers/petsc_compile_run_mcp_server.py"
            )
            # await self.client.connect_to_remote_server(sys.argv[1])
            print("Green agent: Starting code generation request...")

            res = await agent.run(context.message, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {e}", context_id=context_id, task_id=task.id
                )
            )
        finally:
            await self.client.cleanup()

        print("Green agent: Code generation request complete.")
        await event_queue.enqueue_event(new_agent_text_message(f"Finished. ✅\n"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError
