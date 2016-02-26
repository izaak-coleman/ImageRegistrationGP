#include <iostream>
//#include <chrono>

using namespace std;
//using namespace std::chrono;

void quickSort(int arr[], int left, int right);

int main() 
{
    int array[] = {18,5,67,342,234,55,5,2,225,6,7,746,6,32,532,6,76,4,647};
    
    //high_resolution_clock::time_point t1 = high_resolution_clock::now();    
    for (int j=0; j<100000; j++) {
      quickSort(array, 0, 18);
      //high_resolution_clock::time_point t2 = high_resolution_clock::now();
      
      for (int i = 0; i < 19; i++)
	{
	  //cout << array[i] << " ";
	}
    }
    //auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    //cout << endl << duration << endl;
    
    return 0;  
}

void quickSort(int arr[], int left, int right) 
{
    int i = left, j = right;
    int tmp;
    int pivot = arr[(left + right) / 2];
           
    while (i <= j) 
    {
	while (arr[i] < pivot)
	    i++;
	while (arr[j] > pivot)
	    j--;
	
	if (i <= j) 
	{
	    tmp = arr[i];
	    arr[i] = arr[j];
	    arr[j] = tmp;
	    i++; 
	    j--;
	}
    }

    if (left < j)
	quickSort(arr, left, j);   
    if (i < right)
	quickSort(arr, i, right);
}
