#include<iostream>
using namespace std;

class RationalNumber
{
    private:
    int numerator;
    int denominator;
    
    public:
    RationalNumber(int num, int denom)
    {
        if(denom == 0)
        {
            denominator = 1;
            numerator = num;
        }
        else if(denom < 0)
        {
            denominator  = denom * -1;
            numerator = num * -1;
        }
        else
        {
            numerator = num;
            denominator = denom;
        }
    }
    
    int getNumerator()
    {
        return numerator;
    }
    int getDenominator()
    {
        return denominator;
    }
    
    RationalNumber reduce(RationalNumber ret)
    {
        int numerator = ret.getNumerator();
        int denominator = ret.getDenominator();
        
        for (int i = denominator * numerator; i > 1; i--)
        {  
                if ((denominator % i == 0) && (numerator % i == 0))
                {  
                    denominator /= i;
                    numerator /= i;
                }
        }
        RationalNumber result(numerator, denominator);
        return result;
    }
    RationalNumber operator+(RationalNumber f)
    {
        int resultNumerator = (numerator*f.getDenominator()) + (denominator*f.getNumerator());
        int resultDenominator = denominator*f.getDenominator();
        RationalNumber result(resultNumerator, resultDenominator);
        return reduce(result);
    }
    RationalNumber operator-(RationalNumber f)
    {
        
        int resultNumerator = (numerator*f.getDenominator()) - (denominator*f.getNumerator());
        int resultDenominator = denominator*f.getDenominator();
        RationalNumber result(resultNumerator, resultDenominator);
        return reduce(result);
    }
    RationalNumber operator*(RationalNumber f)
    {
        int resultNumerator = numerator*f.getNumerator();
        int resultDenominator = denominator*f.getDenominator();
        RationalNumber result(resultNumerator, resultDenominator);
        return reduce(result);
    }
    RationalNumber operator/(RationalNumber f)
    {
        int resultNumerator = numerator*f.getDenominator();
        int resultDenominator = denominator*f.getNumerator();
        RationalNumber result(resultNumerator, resultDenominator);
        return reduce(result);
    }
    Boolean operator==(RationalNumber f)
    {
        if(numerator == f.getNumerator() && denominator == f.getDenominator())
        return true;
        else
        return false;
    }
    void display()
    {
        cout << numerator << " / " << denominator;
    }
};

int main()
{
    RationalNumber rat1(1,2);
    RationalNumber rat2(1,4);
    RationalNumber test = rat1 * rat2;
    test.display();
    return 0;
}