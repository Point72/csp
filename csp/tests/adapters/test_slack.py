import pytest
from datetime import timedelta
from ssl import create_default_context
from unittest.mock import MagicMock, call, patch

import csp
from csp import ts
from csp.adapters.slack import SlackAdapterManager, SlackMessage, mention_user


@csp.node
def hello(msg: ts[SlackMessage]) -> ts[SlackMessage]:
    if csp.ticked(msg):
        text = f"Hello <@{msg.user_id}>!"
        return SlackMessage(
            channel="a new channel",
            # reply in thread
            thread=msg.thread,
            msg=text,
        )


@csp.node
def react(msg: ts[SlackMessage]) -> ts[SlackMessage]:
    if csp.ticked(msg):
        return SlackMessage(
            channel=msg.channel,
            channel_id=msg.channel_id,
            thread=msg.thread,
            reaction="eyes",
        )


@csp.node
def send_fake_message(clientmock: MagicMock, requestmock: MagicMock, am: SlackAdapterManager) -> ts[bool]:
    with csp.alarms():
        a_send = csp.alarm(bool)
    with csp.start():
        csp.schedule_alarm(a_send, timedelta(seconds=1), True)
    if csp.ticked(a_send):
        if a_send:
            am._process_slack_message(clientmock, requestmock)
            csp.schedule_alarm(a_send, timedelta(seconds=1), False)
        else:
            return True


PUBLIC_CHANNEL_MENTION_PAYLOAD = {
    "token": "ABCD",
    "team_id": "EFGH",
    "api_app_id": "HIJK",
    "event": {
        "client_msg_id": "1234-5678",
        "type": "app_mention",
        "text": "<@BOTID> <@USERID> <@USERID2>",
        "user": "USERID",
        "ts": "1.2",
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "tx381",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "user", "user_id": "BOTID"},
                            {"type": "text", "text": " "},
                            {"type": "user", "user_id": "USERID"},
                            {"type": "text", "text": " "},
                            {"type": "user", "user_id": "USERID2"},
                        ],
                    }
                ],
            }
        ],
        "team": "ABCD",
        "channel": "EFGH",
        "event_ts": "1.2",
    },
    "type": "event_callback",
    "event_id": "ABCD",
    "event_time": 1707423091,
    "authorizations": [
        {"enterprise_id": None, "team_id": "ABCD", "user_id": "BOTID", "is_bot": True, "is_enterprise_install": False}
    ],
    "is_ext_shared_channel": False,
    "event_context": "SOMELONGCONTEXT",
}
DIRECT_MESSAGE_PAYLOAD = {
    "token": "ABCD",
    "team_id": "EFGH",
    "context_team_id": "ABCD",
    "context_enterprise_id": None,
    "api_app_id": "HIJK",
    "event": {
        "client_msg_id": "1234-5678",
        "type": "message",
        "text": "test",
        "user": "USERID",
        "ts": "2.1",
        "blocks": [
            {
                "type": "rich_text",
                "block_id": "gB9fq",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": "test"}]}],
            }
        ],
        "team": "ABCD",
        "channel": "EFGH",
        "event_ts": "2.1",
        "channel_type": "im",
    },
    "type": "event_callback",
    "event_id": "ABCD",
    "event_time": 1707423220,
    "authorizations": [
        {"enterprise_id": None, "team_id": "ABCD", "user_id": "BOTID", "is_bot": True, "is_enterprise_install": False}
    ],
    "is_ext_shared_channel": False,
    "event_context": "SOMELONGCONTEXT",
}


class TestSlack:
    def test_slack_tokens(self):
        with pytest.raises(RuntimeError):
            SlackAdapterManager("abc", "def")

    @pytest.mark.parametrize("payload", (PUBLIC_CHANNEL_MENTION_PAYLOAD, DIRECT_MESSAGE_PAYLOAD))
    def test_slack(self, payload):
        with patch("csp.adapters.slack.SocketModeClient") as clientmock:
            # mock out the event from the slack sdk
            reqmock = MagicMock()
            reqmock.type = "events_api"
            reqmock.payload = payload

            # mock out the user/room lookup responses
            mock_user_response = MagicMock(name="users_info_mock")
            mock_user_response.status_code = 200
            mock_user_response.data = {
                "user": {"profile": {"real_name_normalized": "johndoe", "email": "johndoe@some.email"}, "name": "blerg"}
            }
            clientmock.return_value.web_client.users_info.return_value = mock_user_response
            mock_room_response = MagicMock(name="conversations_info_mock")
            mock_room_response.status_code = 200
            mock_room_response.data = {"channel": {"is_im": False, "is_private": True, "name": "a private channel"}}
            clientmock.return_value.web_client.conversations_info.return_value = mock_room_response
            mock_list_response = MagicMock(name="conversations_list_mock")
            mock_list_response.status_code = 200
            mock_list_response.data = {
                "channels": [
                    {"name": "a private channel", "id": "EFGH"},
                    {"name": "a new channel", "id": "new_channel"},
                ]
            }
            clientmock.return_value.web_client.conversations_list.return_value = mock_list_response

            def graph():
                am = SlackAdapterManager("xapp-1-dummy", "xoxb-dummy", ssl=create_default_context())

                # send a fake slack message to the app
                stop = send_fake_message(clientmock, reqmock, am)

                # send a response
                resp = hello(csp.unroll(am.subscribe()))
                am.publish(resp)

                # do a react
                rct = react(csp.unroll(am.subscribe()))
                am.publish(rct)

                csp.add_graph_output("response", resp)
                csp.add_graph_output("react", rct)

                # stop after first messages
                done_flag = (csp.count(stop) + csp.count(resp) + csp.count(rct)) == 3
                csp.stop_engine(stop)

            # run the graph
            resp = csp.run(graph, realtime=True)

            # check outputs
            if payload == PUBLIC_CHANNEL_MENTION_PAYLOAD:
                assert resp["react"]
                assert resp["response"]

                assert resp["react"][0][1] == SlackMessage(
                    channel="a private channel", channel_id="EFGH", reaction="eyes", thread="1.2"
                )
                assert resp["response"][0][1] == SlackMessage(
                    channel="a new channel", msg="Hello <@USERID>!", thread="1.2"
                )
            else:
                assert resp["react"]
                assert resp["response"]

                assert resp["react"][0][1] == SlackMessage(
                    channel="a private channel", channel_id="EFGH", reaction="eyes", thread="2.1"
                )
                assert resp["response"][0][1] == SlackMessage(
                    channel="a new channel", msg="Hello <@USERID>!", thread="2.1"
                )

            # check all inbound mocks got called
            if payload == PUBLIC_CHANNEL_MENTION_PAYLOAD:
                assert clientmock.return_value.web_client.users_info.call_count == 2
            else:
                assert clientmock.return_value.web_client.users_info.call_count == 1
            assert clientmock.return_value.web_client.conversations_info.call_count == 1

            # check all outbound mocks got called
            assert clientmock.return_value.web_client.reactions_add.call_count == 1
            assert clientmock.return_value.web_client.chat_postMessage.call_count == 1

            if payload == PUBLIC_CHANNEL_MENTION_PAYLOAD:
                assert clientmock.return_value.web_client.reactions_add.call_args_list == [
                    call(channel="EFGH", name="eyes", timestamp="1.2")
                ]
                assert clientmock.return_value.web_client.chat_postMessage.call_args_list == [
                    call(channel="new_channel", text="Hello <@USERID>!")
                ]
            else:
                assert clientmock.return_value.web_client.reactions_add.call_args_list == [
                    call(channel="EFGH", name="eyes", timestamp="2.1")
                ]
                assert clientmock.return_value.web_client.chat_postMessage.call_args_list == [
                    call(channel="new_channel", text="Hello <@USERID>!")
                ]

    def test_mention_user(self):
        assert mention_user("ABCD") == "<@ABCD>"
