import threading
from logging import getLogger
from queue import Queue
from ssl import SSLContext
from threading import Thread
from time import sleep
from typing import Dict, List, Optional, TypeVar

import csp
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.outputadapter import OutputAdapter
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.struct import Struct
from csp.impl.types.tstype import ts
from csp.impl.wiring import py_output_adapter_def, py_push_adapter_def

try:
    from slack_sdk.errors import SlackApiError
    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse
    from slack_sdk.web import WebClient

    _HAVE_SLACK_SDK = True
except ImportError:
    _HAVE_SLACK_SDK = False

T = TypeVar("T")
log = getLogger(__file__)


__all__ = ("SlackMessage", "mention_user", "SlackAdapterManager", "SlackInputAdapterImpl", "SlackOutputAdapterImpl")


class SlackMessage(Struct):
    user: str
    user_email: str  # email of the author
    user_id: str  # user id of the author
    tags: List[str]  # list of mentions

    channel: str  # name of channel
    channel_id: str  # id of channel
    channel_type: str  # type of channel, in "message", "public" (app_mention), "private" (app_mention)

    msg: str  # parsed text payload
    reaction: str  # emoji reacts
    thread: str  # thread id, if in thread
    payload: dict  # raw message payload


def mention_user(userid: str) -> str:
    """Convenience method, more difficult to do in symphony but we want slack to be symmetric"""
    return f"<@{userid}>"


class SlackAdapterManager(AdapterManagerImpl):
    def __init__(self, app_token: str, bot_token: str, ssl: Optional[SSLContext] = None):
        if not _HAVE_SLACK_SDK:
            raise RuntimeError("Could not find slack-sdk installation")
        if not app_token.startswith("xapp-") or not bot_token.startswith("xoxb-"):
            raise RuntimeError("Slack app token or bot token looks malformed")

        self._slack_client = SocketModeClient(
            app_token=app_token,
            web_client=WebClient(token=bot_token, ssl=ssl),
        )
        self._slack_client.socket_mode_request_listeners.append(self._process_slack_message)

        # down stream edges
        self._subscribers = []
        self._publishers = []

        # message queues
        self._inqueue: Queue[SlackMessage] = Queue()
        self._outqueue: Queue[SlackMessage] = Queue()

        # handler thread
        self._running: bool = False
        self._thread: Thread = None

        # lookups for mentions and redirection
        self._room_id_to_room_name: Dict[str, str] = {}
        self._room_id_to_room_type: Dict[str, str] = {}
        self._room_name_to_room_id: Dict[str, str] = {}
        self._user_id_to_user_name: Dict[str, str] = {}
        self._user_id_to_user_email: Dict[str, str] = {}
        self._user_name_to_user_id: Dict[str, str] = {}
        self._user_email_to_user_id: Dict[str, str] = {}

    def subscribe(self):
        return _slack_input_adapter(self, push_mode=csp.PushMode.NON_COLLAPSING)

    def publish(self, msg: ts[SlackMessage]):
        return _slack_output_adapter(self, msg)

    def _create(self, engine, memo):
        # We'll avoid having a second class and make our AdapterManager and AdapterManagerImpl the same
        super().__init__(engine)
        return self

    def start(self, starttime, endtime):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._running:
            self._running = False
            self._slack_client.close()
            self._thread.join()

    def register_subscriber(self, adapter):
        if adapter not in self._subscribers:
            self._subscribers.append(adapter)

    def register_publisher(self, adapter):
        if adapter not in self._publishers:
            self._publishers.append(adapter)

    def _get_user_from_id(self, user_id):
        # try to pull from cache
        name = self._user_id_to_user_name.get(user_id, None)
        email = self._user_id_to_user_email.get(user_id, None)

        # if none, refresh data via web client
        if name is None or email is None:
            ret = self._slack_client.web_client.users_info(user=user_id)
            if ret.status_code == 200:
                # TODO OAuth scopes required
                name = ret.data["user"]["profile"].get("real_name_normalized", ret.data["user"]["name"])
                email = ret.data["user"]["profile"]["email"]
                self._user_id_to_user_name[user_id] = name
                self._user_name_to_user_id[name] = user_id  # TODO is this 1-1 in slack?
                self._user_id_to_user_email[user_id] = email
                self._user_email_to_user_id[email] = user_id
        return name, email

    def _get_user_from_name(self, user_name):
        # try to pull from cache
        user_id = self._user_name_to_user_id.get(user_name, None)

        # if none, refresh data via web client
        if user_id is None:
            # unfortunately the reverse lookup is not super nice...
            # we need to pull all users and build the reverse mapping
            ret = self._slack_client.web_client.users_list()
            if ret.status_code == 200:
                # TODO OAuth scopes required
                for user in ret.data["members"]:
                    name = user["profile"].get("real_name_normalized", user["name"])
                    user_id = user["profile"]["id"]
                    email = user["profile"]["email"]
                    self._user_id_to_user_name[user_id] = name
                    self._user_name_to_user_id[name] = user_id  # TODO is this 1-1 in slack?
                    self._user_id_to_user_email[user_id] = email
                    self._user_email_to_user_id[email] = user_id
            return self._user_name_to_user_id.get(user_name, None)
        return user_id

    def _channel_data_to_channel_kind(self, data) -> str:
        if data.get("is_im", False):
            return "message"
        if data.get("is_private", False):
            return "private"
        return "public"

    def _get_channel_from_id(self, channel_id):
        # try to pull from cache
        name = self._room_id_to_room_name.get(channel_id, None)
        kind = self._room_id_to_room_type.get(channel_id, None)

        # if none, refresh data via web client
        if name is None:
            ret = self._slack_client.web_client.conversations_info(channel=channel_id)
            if ret.status_code == 200:
                # TODO OAuth scopes required
                kind = self._channel_data_to_channel_kind(ret.data["channel"])
                if kind == "message":
                    # TODO use same behavior as symphony adapter
                    name = "DM"
                else:
                    name = ret.data["channel"]["name"]

                self._room_id_to_room_name[channel_id] = name
                self._room_name_to_room_id[name] = channel_id
                self._room_id_to_room_type[channel_id] = kind
        return name, kind

    def _get_channel_from_name(self, channel_name):
        # try to pull from cache
        channel_id = self._room_name_to_room_id.get(channel_name, None)

        # if none, refresh data via web client
        if channel_id is None:
            # unfortunately the reverse lookup is not super nice...
            # we need to pull all channels and build the reverse mapping
            ret = self._slack_client.web_client.conversations_list()
            if ret.status_code == 200:
                # TODO OAuth scopes required
                for channel in ret.data["channels"]:
                    name = channel["name"]
                    channel_id = channel["id"]
                    kind = self._channel_data_to_channel_kind(channel)
                    self._room_id_to_room_name[channel_id] = name
                    self._room_name_to_room_id[name] = channel_id
                    self._room_id_to_room_type[channel_id] = kind
            return self._room_name_to_room_id.get(channel_name, None)
        return channel_id

    def _get_tags_from_message(self, blocks, authorizations=None) -> List[str]:
        """extract tags from message, potentially excluding the bot's own @"""
        authorizations = authorizations or []
        if len(authorizations) > 0:
            bot_id = authorizations[0]["user_id"]  # TODO more than one?
        else:
            bot_id = ""

        tags = []
        to_search = blocks.copy()

        while to_search:
            element = to_search.pop()
            # add subsections
            if element.get("elements", []):
                to_search.extend(element.get("elements"))

            if element.get("type", "") == "user":
                tag_id = element.get("user_id")
                if tag_id != bot_id:
                    # TODO tag with id or with name?
                    name, _ = self._get_user_from_id(tag_id)
                    if name:
                        tags.append(name)
        return tags

    def _process_slack_message(self, client: SocketModeClient, req: SocketModeRequest):
        log.info(req.payload)
        if req.type == "events_api":
            # Acknowledge the request anyway
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)

            if (
                req.payload["event"]["type"] in ("message", "app_mention")
                and req.payload["event"].get("subtype") is None
            ):
                user, user_email = self._get_user_from_id(req.payload["event"]["user"])
                channel, channel_type = self._get_channel_from_id(req.payload["event"]["channel"])
                tags = self._get_tags_from_message(req.payload["event"]["blocks"], req.payload["authorizations"])
                slack_msg = SlackMessage(
                    user=user or "",
                    user_email=user_email or "",
                    user_id=req.payload["event"]["user"],
                    tags=tags,
                    channel=channel or "",
                    channel_id=req.payload["event"]["channel"],
                    channel_type=channel_type or "",
                    msg=req.payload["event"]["text"],
                    reaction="",
                    thread=req.payload["event"]["ts"],
                    payload=req.payload.copy(),
                )
                self._inqueue.put(slack_msg)

    def _run(self):
        self._slack_client.connect()

        while self._running:
            # drain outbound
            while not self._outqueue.empty():
                # pull SlackMessage from queue
                slack_msg = self._outqueue.get()

                # refactor into slack command
                # grab channel or DM
                if hasattr(slack_msg, "channel_id") and slack_msg.channel_id:
                    channel_id = slack_msg.channel_id
                elif hasattr(slack_msg, "channel") and slack_msg.channel:
                    # TODO DM
                    channel_id = self._get_channel_from_name(slack_msg.channel)

                # pull text or reaction
                if (
                    hasattr(slack_msg, "reaction")
                    and slack_msg.reaction
                    and hasattr(slack_msg, "thread")
                    and slack_msg.thread
                ):
                    # TODO
                    self._slack_client.web_client.reactions_add(
                        channel=channel_id,
                        name=slack_msg.reaction,
                        timestamp=slack_msg.thread,
                    )
                elif hasattr(slack_msg, "msg") and slack_msg.msg:
                    try:
                        # send text to channel
                        self._slack_client.web_client.chat_postMessage(
                            channel=channel_id,
                            text=getattr(slack_msg, "msg", ""),
                        )
                    except SlackApiError:
                        # TODO
                        ...
                else:
                    # cannot send empty message, log an error
                    log.error(f"Received malformed SlackMessage instance: {slack_msg}")

            if not self._inqueue.empty():
                # pull all SlackMessages from queue
                # do as burst to match SymphonyAdapter
                slack_msgs = []
                while not self._inqueue.empty():
                    slack_msgs.append(self._inqueue.get())

                # push to all the subscribers
                for adapter in self._subscribers:
                    adapter.push_tick(slack_msgs)

            # do short sleep
            sleep(0.1)

            # liveness check
            if not self._thread.is_alive():
                self._running = False
                self._thread.join()

        # shut down socket client
        try:
            # TODO which one?
            self._slack_client.close()
            # self._slack_client.disconnect()
        except AttributeError:
            # TODO bug in slack sdk causes an exception to be thrown
            #   File "slack_sdk/socket_mode/builtin/connection.py", line 191, in disconnect
            #   self.sock.close()
            #   ^^^^^^^^^^^^^^^
            # AttributeError: 'NoneType' object has no attribute 'close'
            ...

    def _on_tick(self, value):
        self._outqueue.put(value)


class SlackInputAdapterImpl(PushInputAdapter):
    def __init__(self, manager):
        manager.register_subscriber(self)
        super().__init__()


class SlackOutputAdapterImpl(OutputAdapter):
    def __init__(self, manager):
        manager.register_publisher(self)
        self._manager = manager
        super().__init__()

    def on_tick(self, time, value):
        self._manager._on_tick(value)


_slack_input_adapter = py_push_adapter_def(
    name="SlackInputAdapter",
    adapterimpl=SlackInputAdapterImpl,
    out_type=ts[List[SlackMessage]],
    manager_type=SlackAdapterManager,
)
_slack_output_adapter = py_output_adapter_def(
    name="SlackOutputAdapter",
    adapterimpl=SlackOutputAdapterImpl,
    manager_type=SlackAdapterManager,
    input=ts[SlackMessage],
)
