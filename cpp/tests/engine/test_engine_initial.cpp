#include <gtest/gtest.h>
#include <csp/core/Time.h>
#include <csp/core/PullInputAdapter.h>
#include <csp/core/Node.h>
#include <csp/core/Engine.h>

using namespace tcep;

class AddNode : public Node
{
public:
    AddNode( Engine * engine ) : Node( NodeDef( 2, 1 ), engine )
    {
        createOutput<double>( InputId( 0 ) );
    }

    void executeImpl()
    {
        if( input( InputId( 0 ) ) -> valid() &&
            input( InputId( 1 ) ) -> valid() )
        {
            double sum = input( InputId( 0 ) ) -> lastValueTyped<double>() +
                input( InputId( 1 ) ) -> lastValueTyped<double>();
            output( InputId( 0 ) ) -> outputTickTyped<double>( engine() -> time(), sum );

            //printf( "RBA Output from adder at time %s\n", engine() -> time().asString().c_str() );
        }
    }
};

class WriteNode : public Node
{
public:
    WriteNode( Engine * engine, std::string tag ) : Node( NodeDef( 1, 0 ), engine ),
                                                    m_tag( tag )
    {
    }

    void execute()
    {
        //std::cerr << engine() -> time() << ": " << m_tag << " " << input( InputId( 0 ) ) -> lastValueTyped<double>() << std::endl;
        printf( "%s: %s %f\n", engine() -> time().asString().c_str(), m_tag.c_str(), input( InputId( 0 ) ) -> lastValueTyped<double>() );
    }

    std::string m_tag;
};

class DummyPullAdapter : public PullInputAdapter<double>
{
public:
    DummyPullAdapter( Engine * engine, DateTime startTime,
                      TimeDelta interval,
                      int tickCount ) : PullInputAdapter<double>( engine ),
                                        m_startTime( startTime ),
                                        m_interval( interval ),
                                        m_tickCount( tickCount ),
                                        m_curTime( DateTime::NONE() )
    {
        initBuffer<double>();
    }

    /*bool advance() override
    {
        if( count() == m_tickCount )
            return false;
        
        if( m_curTime == DateTime::NONE() )
            m_curTime = m_startTime;
        else
            m_curTime += m_interval;
    }

    DateTime time() override { return m_curTime; }
    void consumeData() override
    {
        consumeDataTyped<double>( m_curTime, count() );
        }*/

    bool next( DateTime & t, double & value )
    {
        if( count() == m_tickCount )
            return false;
        if( m_curTime == DateTime::NONE() )
            m_curTime = m_startTime;
        else
            m_curTime += m_interval;

        t = m_curTime;
        value = count();
        return true;
    }
    
private:
    DateTime  m_startTime;
    TimeDelta m_interval;
    int       m_tickCount;
    DateTime  m_curTime;
};

TEST( EngineTest, test_engine_initial )
{
    Engine engine;
    auto a1 = new DummyPullAdapter( &engine, DateTime( 2017, 1, 1 ), TimeDelta::fromSeconds( 1 ), 20 );
    auto a2 = new DummyPullAdapter( &engine, DateTime( 2017, 1, 1, 0, 0, 1 ), TimeDelta::fromSeconds( 2 ), 20 );

    engine.registerInputAdapter( a1 );
    engine.registerInputAdapter( a2 );
    
    AddNode * addNode = new AddNode( &engine );
    a1 -> linkTo( addNode, InputId( 0 ) );
    a2 -> linkTo( addNode, InputId( 1 ) );

    WriteNode * writeNode = new WriteNode( &engine, "TEST" );
    addNode -> linkTo( InputId( 0 ), writeNode, InputId( 0 ) );

    WriteNode * writeNode2 = new WriteNode( &engine, "A1" );
    a1 -> linkTo( writeNode2, InputId( 0 ) );

    WriteNode * writeNode3 = new WriteNode( &engine, "A2" );
    a2 -> linkTo( writeNode3, InputId( 0 ) );

    addNode -> setRank( 0 );
    writeNode -> setRank( 1 );
    writeNode2 -> setRank( 0 );
    writeNode3 -> setRank( 0 );

    engine.run();
}

/*template<typename ...Args>
struct WiringNode
{

    
};

TEST( EngineTest, test_engine_wiring )
{
    Context context;
    auto a1 = DummyPullAdapterW( DateTime( 2017, 1, 1 ), TimeDelta::fromSeconds( 1 ), 20 );
    auto a2 = DummyPullAdapterW( DateTime( 2017, 1, 1, 0, 0, 1 ), TimeDelta::fromSeconds( 2 ), 20 );

    auto adder  = AddNodeW( a1, a2 );
    auto adder2 = AddNodeW( adder, adder );

    WriteNodeW( "ADDER",  adder );
    WriteNodeW( "ADDER2", adder2 );
    WriteNodeW( "A1",     a1 );
    WriteNodeW( "A2",     a2 );

    std::unique_ptr<Engine> engine( context.createEngine() );
    engine -> run();
};
*/
