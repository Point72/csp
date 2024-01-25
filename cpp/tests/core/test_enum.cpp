#include <csp/core/Enum.h>
#include <gtest/gtest.h>
#include <csp/engine/Enums.h>

using namespace csp;

struct TestEnumTraits
{
    enum _enum : unsigned char
    {
        UNKNOWN = 0,
        A,
        B,
        C,
        F,

        NUM_TYPES
    };

    _enum m_value;
};

using TestEnum = Enum<TestEnumTraits>;

INIT_CSP_ENUM(TestEnum,
    "UNKNOWN",
    "A",
    "B",
    "C",
    "F"
);

TEST( EnumTest, basic_functionality )
{
    ASSERT_EQ( TestEnum::A, TestEnum::A );
    ASSERT_EQ( TestEnum::A, TestEnum( TestEnum::A ) );
    ASSERT_EQ( TestEnum( TestEnum::A ), TestEnum::A );
    ASSERT_EQ( TestEnum::A, TestEnum( 1 ) );

    ASSERT_NE( TestEnum::A, TestEnum::B );
    ASSERT_NE( TestEnum::A, TestEnum( TestEnum::B ) );
    ASSERT_NE( TestEnum( TestEnum::B ), TestEnum::A );
    ASSERT_NE( TestEnum::A, TestEnum( 2 ) );


    ASSERT_EQ( TestEnum( 1 ), TestEnum::A );
    ASSERT_EQ( TestEnum( 1 ).asString(), "A" );
    ASSERT_EQ( TestEnum( 2 ).asString(), "B" );

    ASSERT_EQ( TestEnum( "A" ), TestEnum::A );
    ASSERT_EQ( TestEnum( "B" ), TestEnum::B );
    ASSERT_EQ( TestEnum( "F" ), TestEnum::F );
    ASSERT_EQ( TestEnum( std::string( "F" ) ), TestEnum::F );

    ASSERT_EQ( TestEnum( TestEnum( TestEnum::F ).asString() ), TestEnum( TestEnum::F ) );

    std::stringstream oss;
    oss << TestEnum( "UNKNOWN" );
    ASSERT_EQ( oss.str(), "UNKNOWN");

    ASSERT_THROW( TestEnum( "FOO" ), ValueError );
    ASSERT_THROW( TestEnum( 23 ), ValueError );
}

TEST( EnumTest, iteration )
{
    std::set<std::string> out;
    for( auto it = TestEnum::begin(); it != TestEnum::end(); ++it )
    {
        out.insert( (*it).asString() );
    }

    decltype( out ) comp{ "UNKNOWN", "A", "B" ,"C", "F" };
    ASSERT_EQ( out, comp );
}

TEST( EnumTest, test_external_init )
{
    //was a problem with static linking
    ASSERT_EQ( PushMode( "LAST_VALUE" ), PushMode::LAST_VALUE );
    ASSERT_EQ( PushMode( PushMode::LAST_VALUE ).asString(), "LAST_VALUE" );    
}
