namespace Ametrin.Numerics.Calc;

public interface IFunctionPart
{

}

public interface IScalarPart;

public sealed record AddPart(IFunctionPart Left, IFunctionPart Right) : IFunctionPart;

public sealed record TensorPart(Identifier Name, IndexDescriptor Size) : IFunctionPart
{
    public int Rank => Size.Rank;
}

public sealed record CellScalarPart(TensorPart Tensor, IndexDescriptor Index) : IScalarPart;
public sealed record VariableScalarPart(Identifier Name) : IScalarPart;
public sealed record ConstantScalarPart(decimal Value) : IScalarPart;
public sealed record Identifier(ImmutableList<string> Names);

public sealed record IndexDescriptor(ImmutableArray<IScalarPart> Dimensions)
{
    public int Rank => Dimensions.Length;
    public IndexDescriptor Resolve() => this;
}