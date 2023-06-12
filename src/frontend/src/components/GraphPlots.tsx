export default function GraphPlots({ data, level }: any) {

    return (
        <div className="flex flex-wrap w-full">
            {allImages.map((image, i: number) => {
                return (
                    <div key={i} className="w-1/4">
                        <h1 className="text-center w-full">Level : {level == 0 ?
                            i + 1
                            : level}</h1>
                        <img
                            className=""
                            src={`data:image/png;base64,${image}`}
                            alt="MyImage"
                        />
                    </div>
                );
            })}
        </div>
    );
}
