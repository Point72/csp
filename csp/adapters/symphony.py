import http.client
import json
import requests
import ssl
import threading
from logging import getLogger
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import csp
from csp import ts
from csp.impl.enum import Enum
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

__all__ = ["SymphonyAdapter", "SymphonyMessage"]

log = getLogger(__file__)


def _sync_create_data_feed(datafeed_create_url: str, header: Dict[str, str]) -> str:
    r = requests.post(
        url=datafeed_create_url,
        headers=header,
    )
    datafeed_id = r.json()["id"]
    log.info(f"created symphony datafeed with id={datafeed_id}")
    return r, datafeed_id


def _client_cert_post(host: str, request_url: str, cert_file: str, key_file: str) -> str:
    request_headers = {"Content-Type": "application/json"}
    request_body_dict = {}

    # Define the client certificate settings for https connection
    context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    # Create a connection to submit HTTP requests
    connection = http.client.HTTPSConnection(host, port=443, context=context)

    # Use connection to submit a HTTP POST request
    connection.request(method="POST", url=request_url, headers=request_headers, body=json.dumps(request_body_dict))

    # Print the HTTP response from the IOT service endpoint
    response = connection.getresponse()

    if response.status != 200:
        raise Exception(
            f"Cannot connect for symphony handshake to https://{host}{request_url}: {response.status}:{response.reason}"
        )
    data = response.read().decode("utf-8")
    return json.loads(data)


def _symphony_session(
    auth_host: str,
    session_auth_path: str,
    key_auth_path: str,
    cert_string: str,
    key_string: str,
) -> Dict[str, str]:
    """Setup symphony session and return the header

    Args:
        auth_host (str): authentication host, like `company-api.symphony.com`
        session_auth_path (str): path to authenticate session, like `/sessionauth/v1/authenticate`
        key_auth_path (str): path to authenticate key, like `/keyauth/v1/authenticate`
        cert_string (str): pem format string of client certificate
        key_string (str): pem format string of client private key
    Returns:
        Dict[str, str]: headers from authentication
    """
    with NamedTemporaryFile(mode="wt", delete=False) as cert_file:
        with NamedTemporaryFile(mode="wt", delete=False) as key_file:
            cert_file.write(cert_string)
            key_file.write(key_string)

    data = _client_cert_post(auth_host, session_auth_path, cert_file.name, key_file.name)
    session_token = data["token"]

    data = _client_cert_post(auth_host, key_auth_path, cert_file.name, key_file.name)
    key_manager_token = data["token"]

    headers = {
        "sessionToken": session_token,
        "keyManagerToken": key_manager_token,
        "Accept": "application/json",
    }
    return headers


class Presence(csp.Enum):
    AVAILABLE = Enum.auto()
    AWAY = Enum.auto()


def send_symphony_message(msg: str, room_id: str, message_create_url: str, header: Dict[str, str]):
    """Wrap message string and send it to symphony"""
    out_json = {
        "message": f"""
        <messageML>
        {msg}
        </messageML>
        """
    }
    url = message_create_url.format(sid=room_id)
    return requests.post(
        url=url,
        json=out_json,
        headers=header,
    )


def _get_room_id(room_name: str, room_search_url: str, header: Dict[str, str]):
    """Given a room name, find its room ID"""
    query = {"query": room_name}
    res = requests.post(
        url=room_search_url,
        json=query,
        headers=header,
    )
    if res and res.status_code == 200:
        res_json = res.json()
        for room in res_json["rooms"]:
            # in theory there could be a room whose name is a subset of another, and so the search could return multiple
            # go through search results to find room with name exactly as given
            name = room.get("roomAttributes", {}).get("name")
            if name and name == room_name:
                id = room.get("roomSystemInfo", {}).get("id")
                if id:
                    return id
        return None  # actually no exact matches, or malformed content from symphony
    else:
        log.error(f"ERROR looking up Symphony room_id for room {room_name}: status {res.status_code} text {res.text}")


def _get_room_name(room_id: str, room_info_url: str, header: Dict[str, str]):
    """Given a room ID, find its name"""
    url = room_info_url.format(room_id=room_id)
    res = requests.get(
        url,
        headers=header,
    )
    if res and res.status_code == 200:
        res_json = res.json()
        name = res_json.get("roomAttributes", {}).get("name")
        if name:
            return name
        log.error(
            f"ERROR: malformed response from Symphony room info call to get name from id {room_id} via url {url}: {res_json}"
        )
    else:
        log.error(
            f"ERROR: failed to query Symphony for room name from id {room_id} via url {url}: code {res.status_code} text {res.text}"
        )


def _get_user_mentions(payload):
    # try to extract user mentions
    user_mentions = []
    try:
        payload_data = json.loads(payload.get("data", "{}"))
        for value in payload_data.values():
            if value["type"] == "com.symphony.user.mention":
                # if its a user mention (the only supported one for now),
                # then grab the payload
                user_id = str(value["id"][0]["value"])
                user_mentions.append(user_id)
    finally:
        return user_mentions


class SymphonyRoomMapper:
    def __init__(self, room_search_url: str, room_info_url: str, header: Dict[str, str]):
        self._name_to_id = {}
        self._id_to_name = {}
        self._room_search_url = room_search_url
        self._room_info_url = room_info_url
        self._header = header
        self._lock = threading.Lock()

    def get_room_id(self, room_name):
        with self._lock:
            if room_name in self._name_to_id:
                return self._name_to_id[room_name]
            else:
                room_id = _get_room_id(room_name, self._room_search_url, self._header)
                self._name_to_id[room_name] = room_id
                self._id_to_name[room_id] = room_name
                return room_id

    def get_room_name(self, room_id):
        with self._lock:
            if room_id in self._id_to_name:
                return self._id_to_name[room_id]
            else:
                room_name = _get_room_name(room_id, self._room_info_url, self._header)
                self._name_to_id[room_name] = room_id
                self._id_to_name[room_id] = room_name
                return room_name

    def set_im_id(self, user, id):
        with self._lock:
            self._id_to_name[id] = user
            self._name_to_id[user] = id


def mention_user(email_or_userid: str = ""):
    if email_or_userid:
        if "@" in str(email_or_userid):
            return f'<mention email="{email_or_userid}" />'
        else:
            return f'<mention uid="{email_or_userid}" />'
    return ""


class SymphonyMessage(csp.Struct):
    user: str
    user_email: str  # email of the author, for mentions
    user_id: str  # uid of the author, for mentions
    tags: [str]  # list of user ids in message, for mentions
    room: str
    msg: str
    form_id: str
    form_values: dict


class SymphonyReaderPushAdapterImpl(PushInputAdapter):
    def __init__(
        self,
        datafeed_create_url: str,
        datafeed_delete_url: str,
        datafeed_read_url: str,
        header: Dict[str, str],
        rooms: set,
        exit_msg: str = "",
        room_mapper: Optional[SymphonyRoomMapper] = None,
    ):
        """Setup Symphony Reader

        Args:
            datafeed_create_url (str): string path to create datafeed, like `https://SYMPHONY_HOST/agent/v5/datafeeds`
            datafeed_delete_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/agent/v5/datafeeds/{{datafeed_id}}`
            datafeed_read_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/agent/v5/datafeeds/{{datafeed_id}}/read`

            room_search_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/pod/v3/room/search`
            room_info_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/pod/v3/room/{{room_id}}/info`

            header (Dict[str, str]): authentication headers

            rooms (set): set of initial rooms for the bot to enter
            exit_msg (str): message to send on shutdown
            room_mapper (SymphonyRoomMapper): convenience object to map rooms that bot dynamically discovers
        """
        self._thread = None
        self._running = False

        # message and datafeed
        self._datafeed_create_url = datafeed_create_url
        self._datafeed_delete_url = datafeed_delete_url
        self._datafeed_read_url = datafeed_read_url

        # auth
        self._header = header

        # rooms to enter by default
        self._rooms = rooms
        self._room_ids = set()
        self._exit_msg = exit_msg
        self._room_mapper = room_mapper

    def start(self, starttime, endtime):
        # get symphony session
        resp, datafeed_id = _sync_create_data_feed(self._datafeed_create_url, self._header)
        if resp.status_code not in (200, 201, 204):
            raise Exception(
                f"ERROR: bad status ({resp.status_code}) from _sync_create_data_feed, cannot start Symphony reader"
            )
        else:
            self._url = self._datafeed_read_url.format(datafeed_id=datafeed_id)
            self._datafeed_delete_url = self._datafeed_delete_url.format(datafeed_id=datafeed_id)

        for room in self._rooms:
            room_id = self._room_mapper.get_room_id(room)
            if not room_id:
                raise Exception(f"ERROR: unable to find Symphony room named {room}")
            self._room_ids.add(room_id)

        # start reader thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._thread.start()

    def stop(self):
        if self._running:
            # in order to unblock current requests.get, send a message to one of the rooms we are listening on
            self._running = False
            if self._datafeed_delete_url is not None:
                resp = requests.delete(url=self._datafeed_delete_url, headers=self._header)
                log.info(f"Deleted datafeed with url={self._datafeed_delete_url}: resp={resp}")
            if self._exit_msg:
                send_symphony_message(self._exit_msg, next(iter(self._room_ids)), self._header)
            self._thread.join()

    def _run(self):
        ack_id = ""
        while self._running:
            resp = requests.post(url=self._url, headers=self._header, json={"ackId": ack_id})
            ret = []
            if resp.status_code == 200:
                msg_json = resp.json()
                if "ackId" in msg_json:
                    ack_id = msg_json["ackId"]
                events = msg_json.get("events", [])
                for m in events:
                    if "type" in m and "payload" in m:
                        if m["type"] == "MESSAGESENT":
                            payload = m.get("payload", {}).get("messageSent", {}).get("message")
                            if payload:
                                payload_stream_id = payload.get("stream", {}).get("streamId")
                                if payload_stream_id and (not self._room_ids or payload_stream_id in self._room_ids):
                                    user = payload.get("user", {}).get("displayName", "USER_ERROR")
                                    user_email = payload.get("user", {}).get("email", "USER_ERROR")
                                    user_id = str(payload.get("user", {}).get("userId", "USER_ERROR"))
                                    user_mentions = _get_user_mentions(payload)
                                    msg = payload.get("message", "MSG_ERROR")

                                    # room name or "IM" for direct message
                                    room_type = payload.get("stream", {}).get("streamType", "ROOM")
                                    if room_type == "ROOM":
                                        room_name = self._room_mapper.get_room_name(payload_stream_id)
                                    elif room_type == "IM":
                                        # register the room name for the user so bot can respond
                                        self._room_mapper.set_im_id(user, payload_stream_id)
                                        room_name = "IM"
                                    else:
                                        room_name = ""

                                    if room_name:
                                        ret.append(
                                            SymphonyMessage(
                                                user=user,
                                                user_email=user_email,
                                                user_id=user_id,
                                                tags=user_mentions,
                                                room=room_name,
                                                msg=msg,
                                            )
                                        )
                        elif m["type"] == "SYMPHONYELEMENTSACTION":
                            payload = m.get("payload").get("symphonyElementsAction", {})
                            payload_stream_id = payload.get("stream", {}).get("streamId")

                            if not payload_stream_id:
                                continue

                            if self._room_ids and payload_stream_id not in self._room_ids:
                                continue

                            user = m.get("initiator", {}).get("user", {}).get("displayName", "USER_ERROR")
                            user_email = m.get("initiator", {}).get("user", {}).get("email", "USER_ERROR")
                            user_id = str(m.get("initiator", {}).get("user", {}).get("userId", "USER_ERROR"))
                            user_mentions = _get_user_mentions(m.get("initiator", {}))
                            form_id = payload.get("formId", "FORM_ID_ERROR")
                            form_values = payload.get("formValues", {})

                            # room name or "IM" for direct message
                            room_type = payload.get("stream", {}).get("streamType", "ROOM")
                            if room_type == "ROOM":
                                room_name = self._room_mapper.get_room_name(payload_stream_id)
                            elif room_type == "IM":
                                # register the room name for the user so bot can respond
                                self._room_mapper.set_im_id(user, payload_stream_id)
                                room_name = "IM"
                            else:
                                room_name = ""

                            if room_name:
                                ret.append(
                                    SymphonyMessage(
                                        user=user,
                                        user_email=user_email,
                                        user_id=user_id,
                                        tags=user_mentions,
                                        room=room_name,
                                        form_id=form_id,
                                        form_values=form_values,
                                    )
                                )

            if ret:
                self.push_tick(ret)


SymphonyReaderPushAdapter = py_push_adapter_def(
    "SymphonyReaderPushAdapter",
    SymphonyReaderPushAdapterImpl,
    ts[[SymphonyMessage]],
    datafeed_create_url=str,
    datafeed_delete_url=str,
    datafeed_read_url=str,
    header=Dict[str, str],
    rooms=set,
    exit_msg=str,
    room_mapper=object,
)


def _send_messages(
    msg_queue: Queue,
    header: Dict[str, str],
    room_mapper: SymphonyRoomMapper,
    message_create_url: str,
):
    """read messages from msg_queue and write to symphony. msg_queue to contain instances of SymphonyMessage, or None to shut down"""
    while True:
        msg = msg_queue.get()
        msg_queue.task_done()
        if not msg:  # send None to kill
            break

        room_id = room_mapper.get_room_id(msg.room)
        if not room_id:
            log.error(f"cannot find id for symphony room {msg.room} found in SymphonyMessage")
        else:
            r = send_symphony_message(msg.msg, room_id, message_create_url, header)
            if r.status_code != 200:
                log.error(f"Cannot send message - symphony server response: {r.status_code} {r.text}")


class SymphonyAdapter:
    def __init__(
        self,
        auth_host: str,
        session_auth_path: str,
        key_auth_path: str,
        message_create_url: str,
        presence_url: str,
        datafeed_create_url: str,
        datafeed_delete_url: str,
        datafeed_read_url: str,
        room_search_url: str,
        room_info_url: str,
        cert_string: str,
        key_string: str,
    ):
        """Setup Symphony Reader

        Args:
            auth_host (str): authentication host, like `company-api.symphony.com`

            session_auth_path (str): path to authenticate session, like `/sessionauth/v1/authenticate`
            key_auth_path (str): path to authenticate key, like `/keyauth/v1/authenticate`

            message_create_url (str): string path to create a message, like `https://SYMPHONY_HOST/agent/v4/stream/{{sid}}/message/create`
            presence_url (str): string path to create a message, like `https://SYMPHONY_HOST/pod/v2/user/presence`
            datafeed_create_url (str): string path to create datafeed, like `https://SYMPHONY_HOST/agent/v5/datafeeds`
            datafeed_delete_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/agent/v5/datafeeds/{{datafeed_id}}`
            datafeed_read_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/agent/v5/datafeeds/{{datafeed_id}}/read`

            room_search_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/pod/v3/room/search`
            room_info_url (str): format-string path to create datafeed, like `https://SYMPHONY_HOST/pod/v3/room/{{room_id}}/info`

            cert_string (str): pem format string of client certificate
            key_string (str): pem format string of client private key
            rooms (set): set of initial rooms for the bot to enter
        """
        self._auth_host = auth_host
        self._session_auth_path = session_auth_path
        self._key_auth_path = key_auth_path
        self._message_create_url = message_create_url
        self._presence_url = presence_url
        self._datafeed_create_url = datafeed_create_url
        self._datafeed_delete_url = datafeed_delete_url
        self._datafeed_read_url = datafeed_read_url
        self._room_search_url = room_search_url
        self._room_info_url = room_info_url
        self._cert_string = cert_string
        self._key_string = key_string
        self._header = _symphony_session(
            self._auth_host, self._session_auth_path, self._key_auth_path, self._cert_string, self._key_string
        )
        self._room_mapper = SymphonyRoomMapper(self._room_search_url, self._room_info_url, self._header)

    @csp.graph
    def subscribe(self, rooms: set = set(), exit_msg: str = "") -> ts[[SymphonyMessage]]:
        return SymphonyReaderPushAdapter(
            datafeed_create_url=self._datafeed_create_url,
            datafeed_delete_url=self._datafeed_delete_url,
            datafeed_read_url=self._datafeed_read_url,
            header=self._header,
            rooms=rooms,
            exit_msg=exit_msg,
            room_mapper=self._room_mapper,
        )

    # take in SymphonyMessage and send to symphony on separate thread
    @csp.node
    def _symphony_write(self, msg: ts[SymphonyMessage]):
        with csp.state():
            s_thread = None
            s_queue = None

        with csp.start():
            s_queue = Queue(maxsize=0)
            s_thread = threading.Thread(
                target=_send_messages, args=(s_queue, self._header, self._room_mapper, self._message_create_url)
            )
            s_thread.start()

        with csp.stop():
            if s_thread:
                s_queue.put(None)  # send a None to tell the writer thread to exit
                s_queue.join()  # wait till the writer thread is done with the queue
                s_thread.join()  # then join with the thread

        if csp.ticked(msg):
            s_queue.put(msg)

    @csp.node
    def _set_presense(self, presence: ts[Presence]):
        ret = requests.post(url=self._presence_url, json={"category": presence.name}, headers=self._header)
        if ret.status_code != 200:
            log.error(f"Cannot set presence - symphony server response: {ret.status_code} {ret.text}")

    @csp.graph
    def publish_presence(self, presence: ts[Presence]):
        self._set_presense(presence=presence)

    @csp.graph
    def publish(self, msg: ts[SymphonyMessage]):
        self._symphony_write(msg=msg)
