namespace Ametrin.Numerics.Test;

public sealed class MatMulTests
{
    [Test]
    public async Task Multiply()
    {
        const int Size = 15; // should include a vectorized part and a software part
        var random = new Random(42);

        var left = Matrix.CreateSquare(Size);
        var right = Matrix.CreateSquare(Size);
        var destination = Matrix.CreateSquare(Size);

        NumericsInitializer.HeUniform(left, random);
        NumericsInitializer.HeUniform(right, random);

        left.TransposeLeftMultiplyAddTo(right, destination);
        var leftTmulResult = destination.ToString();

        destination.ResetZero();
        TransposeLeftMultiplyAddToNaive(left, right, destination);
        var leftTmulExpected = destination.ToString();

        await Assert.That(leftTmulResult).IsEqualTo(leftTmulExpected);

        left.MultiplyTo(right, destination);
        var mulResult = destination.ToString();

        MultiplyToNaive(left, right, destination);
        var mulExpected = destination.ToString();
    
        await Assert.That(mulResult).IsEqualTo(mulExpected);
    }

    private static void MultiplyToNaive(Matrix left, Matrix right, Matrix destination)
    {
        for (var leftRowIndex = 0; leftRowIndex < left.RowCount; leftRowIndex++)
        {
            for (var rightColumnIndex = 0; rightColumnIndex < right.ColumnCount; rightColumnIndex++)
            {
                var sum = 0.0f;
                for (var innerIndex = 0; innerIndex < left.ColumnCount; innerIndex++)
                {
                    sum += left[leftRowIndex, innerIndex] * right[innerIndex, rightColumnIndex];
                }
                destination[leftRowIndex, rightColumnIndex] = sum;
            }
        }
    }

    private static void TransposeLeftMultiplyAddToNaive(Matrix left, Matrix right, Matrix destination)
    {
        for (var innerIndex = 0; innerIndex < left.RowCount; innerIndex++)
        {
            // iteration over columns which become rows through the transpose
            for (var leftRowIndex = 0; leftRowIndex < left.ColumnCount; leftRowIndex++)
            {
                for (var rightColumnIndex = 0; rightColumnIndex < right.ColumnCount; rightColumnIndex++)
                {
                    destination[leftRowIndex, rightColumnIndex] += left[innerIndex, leftRowIndex] * right[innerIndex, rightColumnIndex];
                }
            }
        }
    }
}
