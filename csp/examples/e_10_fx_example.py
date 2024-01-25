import csp
from   csp import ts
import csp.showgraph
from csp.adapters.kafka import KafkaAdapterManager, ProtoMessageMapper
from csp.adapters.perspective import PerspectiveAdapter
from datetime import datetime, timedelta
import os.path

class Timestamp( csp.Struct ):
    seconds: int
    nanos: int

class FxMessage( csp.Struct ):
    # timestamp: Timestamp
    pair:  str
    lp:    str
    side:  str
    tier:  float
    price: float

class MyData( csp.Struct ):
    currency: str
    bidprice: float
    bidtier: float
    asktier: float
    askprice: float

@csp.node
def to_string( x:ts[float]) -> ts[str]:

    if csp.ticked(x):
        return '%.6f' % x

@csp.graph
def invert( inverse_name : str, fx_data : ts[ FxMessage ]) -> ts[FxMessage]:

    flip_side = csp.multiplex({'B': csp.const('S'), 'S': csp.const('B')}, fx_data.side, tick_on_index=True)
    return FxMessage.fromts(pair=csp.const(inverse_name), lp=fx_data.lp, side=flip_side, tier=fx_data.tier, price=csp.const(1.0) / fx_data.price)

@csp.graph
def fx_graph():
    broker = "server:9092"
    kafka = KafkaAdapterManager( broker )
    topic = 'test2'

    field_map = {
        # 't_entry': {
        #     'timestamp': {
        #         'seconds': 'seconds',
        #         'nanos': 'nanos'
        #     }
        # },
        'pair': 'pair',
        'lp'  : 'lp',
        'side': 'side',
        'tier': 'tier',
        'px'  : 'price'
    }

    msg_mapper = ProtoMessageMapper( proto_directory = os.path.dirname(__file__), proto_filename = 'e_10_fx_example.proto', proto_message = 'Snapshot' )

    fx_data = kafka.subscribe( ts_type = FxMessage, msg_mapper = msg_mapper, topic = topic, group_id = 0, field_map = field_map )

    csp.print( 'fx_data', fx_data )

    # demux fat-pipe stream by currency pair
    pairs = [ 'USD/JPY', 'EUR/USD', 'GBP/USD' ]
    demuxed_data = csp.demultiplex( fx_data, fx_data.pair, pairs )

    #Normalize USD/JPY to JPY/USD
    demuxed_data[ 'JPY/USD' ] = invert( 'JPY/USD', demuxed_data[ 'USD/JPY'] )

    #convert individual streams into clean struct of bid/ask price/tier
    all_structs = []
    for pair in demuxed_data.keys():
        side = csp.split(demuxed_data[pair].side == 'B', demuxed_data[pair])
        bid = side.true
        ask = side.false
        data = MyData.fromts( currency = csp.const( str( pair ) ), bidprice = bid.price, bidtier = bid.tier, asktier = ask.tier, askprice = ask.price )
        all_structs.append( data )

    data = csp.flatten( all_structs )
    csp.print( 'data', data )

    # publish out to perspective output adapter
    port = 7178
    perspective_adapter = PerspectiveAdapter(port)
    table = perspective_adapter.create_table('my_table', index='currency')
    table.publish( data )
    table.publish(to_string(data.bidprice), 'bidprice_str')
    table.publish( to_string(data.askprice), 'askprice_str')
    table.publish( to_string(data.askprice-data.bidprice), 'spread_str' )

if __name__ == '__main__':
    show_graph = False
    if show_graph:
        csp.showgraph.show_graph(fx_graph)
    else:
        csp.run( fx_graph, starttime = datetime.utcnow(), endtime = timedelta( seconds = 3600 ), realtime = True )