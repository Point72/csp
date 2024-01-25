import csp
from csp import ts
from csp.adapters.kafka import KafkaAdapterManager, BytesMessageProtoMapper, JSONTextMessageMapper, DateTimeType, RawTextMessageMapper
from datetime import datetime, timedelta
from enum import Enum
import os

class JSONMessage( csp.Struct ):
    text: str
    price: float
    size: int

class Timestamp( csp.Struct ):
    seconds: int
    nanos: int

class ProtoMessage( csp.Struct ):
    timestamp: Timestamp
    pair:  str
    lp:    str
    # side:  Enum
    tier:  float
    price: float

class ProtoMessageMulti( csp.Struct ):
    t_entry: Timestamp
    pair: str
    lp: str
    tier: float
    px: float

@csp.graph
def json_graph():
    broker = "localhost:9092"
    kafka = KafkaAdapterManager( broker )

    topic = 'quickstart-events'

    field_map = { 'text': 'text', 'PRICE': 'price', 'SIZE': 'size' }

    msg_mapper = JSONTextMessageMapper()

    data = kafka.subscribe( ts_type = JSONMessage, msg_mapper = msg_mapper, topic = topic, field_map = field_map )
    csp.print( 'data', data )

class MyData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime
    # date : date
    # td : timedelta

class SubData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime
    b2: bool
    i2: int
    d2: float
    s2: str
    dt2: datetime
    prop1: float
    prop2: str

@csp.node
def curtime(x: ts[object]) -> ts[datetime]:
    if csp.ticked(x):
        return csp.now()

@csp.graph
def json_producer_graph():
    broker = "localhost:9092"
    kafka = KafkaAdapterManager( broker )
    # topic = 'quickstart-events-2'
    # topic = 'mytopic123'
    topic = 'events'

    b = csp.merge(csp.timer(timedelta(seconds=.2), True), csp.delay(csp.timer(timedelta(seconds=.2), False), timedelta(seconds=.1)))
    i = csp.count(csp.timer(timedelta(seconds=.15)))
    d = csp.count(csp.timer(timedelta(seconds=.2))) / 2.0
    s = csp.sample(csp.timer(timedelta(seconds=.4)), csp.const('STRING'))
    dt = curtime(b)
    struct = MyData.collectts(b=b, i=i, d=d, s=s,dt=dt)

    msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)

    struct_field_map = {'b': 'b2', 'i': 'i2', 'd': 'd2', 's': 's2', 'dt' : 'dt2'}

    kafka.publish( msg_mapper = msg_mapper, topic = topic, x = struct, field_map = struct_field_map, key = 'events1' )
    pub_data = SubData.collectts(b=b, i=i, d=d, s=s,dt=dt,
                                 b2=struct.b, i2=struct.i, d2=struct.d, s2=struct.s,dt2=struct.dt)

    csp.print( 'pub_data', pub_data )

    # data = kafka.subscribe( ts_type = JSONMessage, msg_mapper = msg_mapper, topic = topic, group_id = 0, push_mode = csp.PushMode.NON_COLLAPSING )
    sub_data = kafka.subscribe( ts_type = SubData, msg_mapper = msg_mapper, topic = topic, key = 'events1' )

    csp.print( 'sub_data', sub_data )

    status = kafka.status()

    csp.print( 'status', status )

@csp.graph
def proto_graph():
    broker = "localhost:9092"
    kafka = KafkaAdapterManager( broker )

    topic = 'test2'

    field_map = {
        't_entry': {
            'timestamp': {
                'seconds': 'seconds',
                'nanos': 'nanos'
            }
        },
        'pair': 'pair',
        'lp'  : 'lp',
        # 'side': 'side',
        'tier': 'tier',
        'px'  : 'price'
    }

    msg_mapper = ProtoMessageMapper( proto_directory = '/tmp', proto_filename = 'fxspotstream.proto', proto_message = 'Snapshot' )

    data = kafka.subscribe( ts_type = ProtoMessage, msg_mapper = msg_mapper, topic = topic, field_map = field_map, key = 'events' )
    csp.print( 'data', data )

@csp.graph
def proto_graph_multiple_subscribers():
    broker = "localhost:9092"
    kafka = KafkaAdapterManager( broker )

    topic = 'test2'

    msg_mapper = ProtoMessageMapper( proto_directory = '/tmp', proto_filename = 'fxspotstream.proto', proto_message = 'Snapshot' )

    data = kafka.subscribe( ts_type = ProtoMessageMulti, msg_mapper = msg_mapper, topic = topic )
    data2 = kafka.subscribe( ts_type = ProtoMessageMulti, msg_mapper = msg_mapper, topic = topic )

    csp.print( 'data', data )
    csp.print( 'data2', data2 )

class TextMessage( csp.Struct ):
    text: str

@csp.graph
def kerberos_consumer_graph():
    broker = 'testbroker:9093'
    kafka = KafkaAdapterManager( broker, auth = True,
                                 sasl_kerberos_keytab = os.getenv( 'KEYTAB_LOCATION' ),
                                 sasl_kerberos_principal = os.getenv( 'USER_PRINCIPAL_NAME' ),
                                 ssl_ca_location = os.getenv( 'KAFKA_PEM_FILE' ) )
    topic = 'mktdata'

    msg_mapper = JSONTextMessageMapper()
    sub_data = kafka.subscribe( ts_type = TextMessage, msg_mapper = msg_mapper, topic = topic, key = 'events1',
                                reset_offset = 'none' )
    csp.print( 'sub_data', sub_data )

    status = kafka.status()
    csp.print( 'status', status )

@csp.graph
def text_consumer_graph():
    broker = 'testbroker:9093'
    kafka = KafkaAdapterManager( broker, auth = True,
                                 sasl_kerberos_keytab = os.getenv( 'KEYTAB_LOCATION' ),
                                 sasl_kerberos_principal = os.getenv( 'USER_PRINCIPAL_NAME' ),
                                 ssl_ca_location = os.getenv( 'KAFKA_PEM_FILE' ) )
    topic = 'mktdata.aggregation.barra_factor_return'

    field_map = { '' : 'text' }

    msg_mapper = RawTextMessageMapper()
    sub_data = kafka.subscribe( ts_type = TextMessage, msg_mapper = msg_mapper, topic = topic, key = 'barra',
                                field_map = field_map, push_mode = csp.PushMode.NON_COLLAPSING, reset_offset = 'earliest' )
    csp.print( 'sub_data', sub_data )

    status = kafka.status()
    csp.print( 'status', status )

if __name__ == '__main__':
    # csp.run( json_graph, starttime = datetime.utcnow(), endtime = timedelta( seconds = 10 ), realtime = True )
    # csp.run( json_producer_graph, starttime = datetime.utcnow(), endtime = timedelta( seconds = 10 ), realtime = True )
    # csp.run( proto_graph, starttime = datetime.utcnow(), endtime = timedelta( seconds = 10 ), realtime = True )
    # csp.run( proto_graph_multiple_subscribers, starttime = datetime.utcnow(), endtime = timedelta( seconds = 10 ), realtime = True )
    # csp.run( kerberos_consumer_graph, starttime = datetime.utcnow(), endtime = timedelta( seconds = 100 ), realtime = True )
    csp.run( text_consumer_graph, starttime = datetime.utcnow(), endtime = timedelta( seconds = 10 ), realtime = True )
