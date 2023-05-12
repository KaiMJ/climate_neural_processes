import PropTypes from "prop-types";

function PopupTemplate({ children, onClose }) {
  return (
    <div className="fixed inset-0 z-40 flex h-screen content-center justify-center rounded-lg">
      <div
        className="absolute inset-0 box-border  bg-black opacity-50"
        onClick={onClose}
      ></div>
      <div className="z-50 my-auto flex h-min flex-row rounded bg-white">
        {children}
      </div>
    </div>
  );
}

PopupTemplate.propTypes = {
  children: PropTypes.node,
  onClose: PropTypes,
};

export default PopupTemplate;
