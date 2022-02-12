#include <string>
#include <iostream>
#include <iomanip>
using namespace std;
#ifndef COMPLEX_H
#define COMPLEX_H

class Complex
{
public:
	Complex operator+(const Complex&) const; // addition
	Complex operator-(const Complex&) const; // subtraction
	Complex operator*(const Complex&) const; // multiplication
	Complex operator/(const Complex&) const; // division

	explicit Complex(double realPart = 0.0 , double imaginaryPart = 0.0 )
	{
		real = realPart;
		imaginary = imaginaryPart;
	}
	
	Complex conjugate(const Complex& c)
	{
		Complex result(c.real, -1 * c.imaginary);
		return result;
	}
	
	Complex operator+(const Complex& c)
	{
		return  Complex{ real + c.real, imaginary + c.imaginary };
	}
	Complex operator-(const Complex& c)
	{
		return  Complex{ real - c.real, imaginary - c.imaginary };
	}
	Complex operator*(const Complex& c)
	{
		return  Complex{ (real * c.real) - (imaginary * c.imaginary)
						,(real * c.imaginary) + (imaginary * c.real) };

	}

	//Division
	Complex operator/(const Complex& c)
	{
		Complex denominator = conjugate(c);
		double deno = pow(c.real, 2) + pow(c.imaginary, 2);

		return Complex((real * c.real + imaginary * c.imaginary) / deno,
			(imaginary * c.real - real * c.imaginary) / deno);
	}

	friend ostream &operator<<( ostream& output, const Complex &c)
	{
		if (c.real != 0.0 && c.imaginary != 0.0)
		{
			if (c.imaginary > 0)
				output << setprecision(3)<<"(" << c.real << " + " << c.imaginary << "i)";
			else
				output << setprecision(3)<<"(" << c.real << " " << c.imaginary << "i)";
		}
		else
		{
			if (c.real == 0.0 && c.imaginary != 0.0 && c.imaginary != 1)
				output <<"(" << c.imaginary << "i)";
			else if(c.real == 0.0 && c.imaginary == 1)
				output << "i";
			else if (c.real != 0.0 && c.imaginary == 0.0)
				output << c.real;
			else
				output << "(0 + 0i)";
		}

		 return output;
	}
	 friend istream &operator>>(istream& input, Complex &c)
	{
		input >> c.real >> c.imaginary;
		return input;
	}

	bool operator==(const Complex& c)
	{
		if (real == c.real && imaginary == c.imaginary)
			return true;
		else
			return false;
	}
	bool operator!=(const Complex& c)
	{
		if (real == c.real && imaginary == c.imaginary)
			return false;
		else
			return true;
	}
private:
	double real;
	double imaginary;
};

#endif // !COMPLEX_H

int main()
{
	string ans;
	bool cond = false;
	Complex x,y,z;
	/*
	Complex 
		a{ 0,1 },
		b(2, 3),
		c(-4,5),
		d(4, 0),
		e(2.5, 3.33);

	cout << "The complex number z : " << z << endl;
	cout << "The complex number a : " << a << endl;
	cout << "The complex number b : " << b << endl;
	cout << "The complex number c : " << c << endl;
	cout << "The complex number d : " << d << endl;
	cout << "The complex number e : " << e << endl;

	z = a + b;
	cout << "\nz = a + b:\n" << z << " = " << a << " + " << b << endl;
	z = b - c;
	cout << "\nz = b - c:\n" << z << " = " << b << " - " << c << endl;
	z = c * d;
	cout << "\nz = c * d:\n" << z << " = " << c << " * " << d << endl;
	z = d / e;
	cout << "\nz = d / e:\n" << z << " = " << d << " / " << e << endl;
	ans = (b == e) ? "True" : "False";
	cout << "\nb == e: " << ans;
	ans = (d != e) ? "True" : "False";
	cout << "\nd != e: " << ans << endl;
	ans = (c == c) ? "True" : "False";
	cout << "\nc == c: " << ans;
	ans = (c != c) ? "True" : "False";
	cout << "\nc != c: " << ans;

	cout << endl;
	*/
	while (!cond)
	{
		Complex z( 0 ,0 );
		cout << "Enter the complex number x (real before imaginary): ";
		cin >> x; 
		cout << "Enter the complex number y (real before imaginary): ";
		cin >> y;

		cout << "The complex number x : " << x << endl;
		cout << "The complex number y : " << y << endl;
		cout << "The complex number z : " << z << endl;

		z = x + y;
		cout << "\nz = x + y:\n" << z << " = " << x <<" + " << y <<endl;
		z = x - y;
		cout << "\nz = x - y:\n" << z << " = " << x << " - " << y << endl;
		z = x * y;
		cout << "\nz = x * y:\n" << z << " = " << x << " * " << y << endl;
		z = x / y;
		cout << "\nz = x / y:\n" << z << " = " << x << " / " << y <<endl;

		ans = (x == y) ? "True" : "False";
		cout << "\nx == y: " <<ans;
	
		ans = (x != y) ? "True" : "False";
		cout << "\nx != y: " << ans << endl;

		cout << "\nDo you want to continue (y/n)? ";
		cin.clear(); cin.ignore(10000, '\n');
		cin >> ans;
		
		if (ans == "y")
			cond = false;
		else if (ans == "n")
			cond = true;
		cout << endl;
	}

	system("pause");
	return 0;
}
