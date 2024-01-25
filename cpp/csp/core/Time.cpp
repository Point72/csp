#include <csp/core/Time.h>

namespace csp
{

// XXXXXX
//This Code was taken from glibc time/offtime.c, the reason being that gmtime_r actually calls the global tzlock
//(even though technically we dont need tz here ).  This makes it lockless
static const unsigned short int __mon_yday[2][13] =
  {
    /* Normal years.  */
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
    /* Leap years.  */
    { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
  };


tm DateTime::asTM() const
{
    tm TM{0};
    time_t days, rem, y;
    const unsigned short int *ip;

    days = m_ticks / NANOS_PER_DAY;
    rem = m_ticks % NANOS_PER_DAY;
    while (rem < 0)
    {
        rem += NANOS_PER_DAY;
        --days;
    }
    while (rem >= NANOS_PER_DAY)
    {
        rem -= NANOS_PER_DAY;
        ++days;
    }
    TM.tm_hour = rem / ( 3600 * NANOS_PER_SECOND );
    rem %= 3600 * NANOS_PER_SECOND;
    TM.tm_min = rem / ( 60 * NANOS_PER_SECOND );
    TM.tm_sec = ( rem % ( 60 * NANOS_PER_SECOND ) ) / NANOS_PER_SECOND;
    /* January 1, 1970 was a Thursday.  */
    TM.tm_wday = (4 + days) % 7;
    if (TM.tm_wday < 0)
        TM.tm_wday += 7;
    y = 1970;

#define DIV(a, b) ((a) / (b) - ((a) % (b) < 0))
#define LEAPS_THRU_END_OF(y) (DIV (y, 4) - DIV (y, 100) + DIV (y, 400))
#define __isleap(year) \
  ((year) % 4 == 0 && ((year) % 100 != 0 || (year) % 400 == 0))

    while (days < 0 || days >= (__isleap(y) ? 366 : 365))
    {
        /* Guess a corrected year, assuming 365 days per year.  */
        time_t yg = y + days / 365 - (days % 365 < 0);

        /* Adjust DAYS and Y to match the guessed year.  */
        days -= ((yg - y) * 365
                 + LEAPS_THRU_END_OF (yg - 1)
                 - LEAPS_THRU_END_OF (y - 1));
        y = yg;
    }
    TM.tm_year = y - 1900;
    if (TM.tm_year != y - 1900)
        CSP_THROW( RuntimeException, "Failed to convert DateTime to struct tm: year overflow" );

    TM.tm_yday = days;
    ip = __mon_yday[__isleap(y)];
    for (y = 11; days < (long int) ip[y]; --y)
        continue;
    days -= ip[y];
    TM.tm_mon = y;
    TM.tm_mday = days + 1;
    return TM;
}
//XXXX

};
