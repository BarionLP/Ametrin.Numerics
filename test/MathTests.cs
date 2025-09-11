namespace Ametrin.Numerics.Test;
public sealed class MathTests
{
    [Test]
    public async Task PointwiseMultiplyAdd()
    {
        var destination = Vector.Of([1, 1, 1, 1, 1, 1, 1, 1, 1]);
        await Assert.That(destination.Count).IsEqualTo(9);

        var left = Vector.Of([2, 2, 2, 2, 2, 2, 2, 2, 2]);
        var right = Vector.Of([2, 2, 2, 2, 2, 2, 2, 2, 2]);

        left.PointwiseMultiplyAddTo(right, destination);
        await Assert.That(destination.Sum()).IsEqualTo(destination.Count * 5);
    }
}
